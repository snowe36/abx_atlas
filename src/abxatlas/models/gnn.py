"""From-scratch graph neural network for Gram-negative QSAR (GPU-trained).

Mirrors the SplitResult contract used by sklearn baselines in qsar.py so it
merges seamlessly into the same leakage-benchmark CSV/plot.

This module imports torch / torch_geometric at module scope (part of the
optional `gpu` extra). It is only ever imported lazily by run.py when
--with-gnn is requested, so plain CPU installs never pay this import cost.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GINConv, global_max_pool, global_mean_pool

from abxatlas.config import RANDOM_STATE
from abxatlas.models.qsar import SplitResult

logger = logging.getLogger(__name__)

DEFAULT_GNN_CONFIG: dict[str, Any] = {
    "hidden_dim": 64,
    "num_layers": 3,
    "dropout": 0.2,
    "conv_type": "gcn",
    "lr": 1e-3,
    "weight_decay": 1e-5,
}


class GNNClassifier(nn.Module):
    """Small message-passing classifier: GCN/GIN stack + mean+max pooling + MLP head."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        conv_type: str = "gcn",
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            self.convs.append(self._make_conv(conv_type, dims[i], dims[i + 1]))
            self.bns.append(nn.BatchNorm1d(dims[i + 1]))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _make_conv(conv_type: str, in_dim: int, out_dim: int):
        if conv_type == "gin":
            mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))
            return GINConv(mlp)
        return GCNConv(in_dim, out_dim)

    def forward(self, x, edge_index, batch):
        h = x
        for conv, bn in zip(self.convs, self.bns, strict=True):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        pooled = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)
        return self.head(pooled).squeeze(-1)


def _device(device: str | None = None) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def train_gnn(
    train_graphs: list,
    y_train: np.ndarray,
    *,
    hidden_dim: int = 64,
    num_layers: int = 3,
    dropout: float = 0.2,
    conv_type: str = "gcn",
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 60,
    batch_size: int = 64,
    val_fraction: float = 0.15,
    patience: int = 12,
    device: str | None = None,
    random_state: int = RANDOM_STATE,
    verbose: bool = False,
) -> tuple[GNNClassifier, str]:
    """Train a GNNClassifier; returns (model, device).

    Early-stops on an internal validation slice carved out of train_graphs
    (never touches the outer test/held-out scaffolds).
    """
    device = _device(device)
    y = np.asarray(y_train, dtype=np.float32)
    # Clone before attaching labels — train_graphs are shared Data objects
    # (slices of a cached graph list reused across splits/HPO trials), and
    # mutating them in place would leak `.y` across unrelated evaluations.
    graphs = [g.clone() for g in train_graphs]
    for g, label in zip(graphs, y, strict=True):
        g.y = torch.tensor([float(label)])

    rng = np.random.RandomState(random_state)
    n = len(graphs)
    idx = rng.permutation(n)
    n_val = int(round(n * val_fraction)) if n >= 40 else 0
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    train_subset = [graphs[i] for i in tr_idx]
    val_subset = [graphs[i] for i in val_idx] if n_val else []

    in_dim = graphs[0].x.shape[1]
    model = GNNClassifier(
        in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        conv_type=conv_type,
    ).to(device)

    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size) if val_subset else None

    best_state = None
    best_val = -float("inf")
    best_epoch = 0
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y.float())
            loss.backward()
            optimizer.step()

        if val_loader is not None:
            val_auc = _eval_auc(model, val_loader, device)
            if verbose:
                logger.info("epoch %d val_auc=%.4f", epoch, val_auc)
            if val_auc > best_val:
                best_val = val_auc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
            elif epoch - best_epoch > patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, device


def _eval_auc(model: GNNClassifier, loader: DataLoader, device: str) -> float:
    proba, y = _predict_loader(model, loader, device)
    if len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, proba))


def _predict_loader(model: GNNClassifier, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            ys.append(batch.y.cpu().numpy())
    return np.concatenate(probs), np.concatenate(ys)


def predict_gnn(model: GNNClassifier, graphs: list, device: str, batch_size: int = 128) -> np.ndarray:
    model.eval()
    loader = DataLoader(list(graphs), batch_size=batch_size, shuffle=False)
    probs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs) if probs else np.array([])


def evaluate_gnn_split(
    graphs: list,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    split_name: str,
    model_name: str = "gnn",
    config: dict[str, Any] | None = None,
    epochs: int = 60,
    random_state: int = RANDOM_STATE,
) -> list[SplitResult]:
    """Train + evaluate a GNN on one leakage split; SplitResult-compatible."""
    y = np.asarray(y)
    if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
        logger.warning("Skipping GNN split %s: need both classes in train and test", split_name)
        return []

    cfg = {**DEFAULT_GNN_CONFIG, **(config or {})}
    train_graphs = [graphs[i] for i in train_idx]
    test_graphs = [graphs[i] for i in test_idx]

    model, device = train_gnn(
        train_graphs,
        y[train_idx],
        epochs=epochs,
        random_state=random_state,
        **cfg,
    )
    proba = predict_gnn(model, test_graphs, device)
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
