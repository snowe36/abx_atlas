"""Fine-tuned pretrained chemical-language transformer (ChemBERTa-style) for
Gram-negative QSAR (GPU fine-tuning).

Mirrors the SplitResult contract used by the sklearn baselines / GNN so it
merges into the same leakage-benchmark CSV/plot. Imports transformers/torch
at module scope (part of the optional `gpu` extra) — safe since this module
is only ever imported lazily by run.py when --with-pretrained is requested.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from abxatlas.config import RANDOM_STATE
from abxatlas.models.qsar import SplitResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
DEFAULT_PRETRAINED_CONFIG: dict[str, Any] = {
    "lr": 2e-5,
    "epochs": 3,
    "dropout": 0.1,
    "max_length": 128,
    "batch_size": 32,
}


class SmilesDataset(Dataset):
    """Tokenizes SMILES lazily; kept simple since datasets here are well under 100k rows."""

    def __init__(
        self,
        smiles: Sequence[str],
        labels: np.ndarray | None,
        tokenizer,
        max_length: int = 128,
    ) -> None:
        self.smiles = list(smiles)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.smiles[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


def _device(device: str | None = None) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def fine_tune_pretrained(
    smiles_train: Sequence[str],
    y_train: np.ndarray,
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    lr: float = 2e-5,
    epochs: int = 3,
    dropout: float = 0.1,
    max_length: int = 128,
    batch_size: int = 32,
    device: str | None = None,
    verbose: bool = False,
) -> tuple[Any, Any, str]:
    """Fine-tune a pretrained SMILES transformer for binary classification.

    Returns (model, tokenizer, device). Meant for GPU (pod); also runs on CPU
    with a tiny epoch count for local smoke tests.
    """
    device = _device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, hidden_dropout_prob=dropout
    ).to(device)

    y = np.asarray(y_train, dtype=np.float32)
    dataset = SmilesDataset(smiles_train, y, tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            optimizer.zero_grad()
            out = model(**batch)
            logits = out.logits.squeeze(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        if verbose:
            logger.info("epoch %d loss=%.4f", epoch, total_loss / max(len(loader), 1))
    return model, tokenizer, device


def predict_pretrained(
    model,
    tokenizer,
    smiles: Sequence[str],
    device: str,
    max_length: int = 128,
    batch_size: int = 64,
) -> np.ndarray:
    dataset = SmilesDataset(smiles, None, tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            logits = out.logits.squeeze(-1)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs) if probs else np.array([])


def evaluate_pretrained_split(
    smiles: Sequence[str],
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    split_name: str,
    model_name: str = "chemberta",
    model_name_hf: str = DEFAULT_MODEL_NAME,
    config: dict[str, Any] | None = None,
    random_state: int = RANDOM_STATE,
) -> list[SplitResult]:
    """Fine-tune + evaluate the pretrained transformer on one leakage split."""
    y = np.asarray(y)
    if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
        logger.warning(
            "Skipping pretrained split %s: need both classes in train and test", split_name
        )
        return []

    cfg = {**DEFAULT_PRETRAINED_CONFIG, **(config or {})}
    smiles = list(smiles)
    smiles_train = [smiles[i] for i in train_idx]
    smiles_test = [smiles[i] for i in test_idx]

    torch.manual_seed(random_state)
    model, tokenizer, device = fine_tune_pretrained(
        smiles_train,
        y[train_idx],
        model_name=model_name_hf,
        lr=cfg["lr"],
        epochs=cfg["epochs"],
        dropout=cfg["dropout"],
        max_length=cfg["max_length"],
        batch_size=cfg["batch_size"],
    )
    proba = predict_pretrained(model, tokenizer, smiles_test, device, max_length=cfg["max_length"])
    pred = (proba >= 0.5).astype(int)
    y_test = y[test_idx]
    return [
        SplitResult(
            split_name=split_name,
            model_name=model_name,
            n_train=int(len(train_idx)),
            n_test=int(len(test_idx)),
            roc_auc=float(roc_auc_score(y_test, proba)),
            average_precision=float(average_precision_score(y_test, proba)),
            balanced_accuracy=float(balanced_accuracy_score(y_test, pred)),
        )
    ]
