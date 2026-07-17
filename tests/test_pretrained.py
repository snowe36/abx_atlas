"""Unit tests for the pretrained-transformer data prep only.

Deliberately avoids downloading real model/tokenizer weights (slow, network-
dependent) — the actual fine-tuning path (fine_tune_pretrained /
evaluate_pretrained_split) is exercised on the RunPod GPU pod, not here.
"""

import numpy as np
import pytest

pytest.importorskip("transformers")

import torch  # noqa: E402

from abxatlas.models.pretrained import SmilesDataset  # noqa: E402


class _FakeTokenizer:
    """Minimal duck-typed stand-in for a HuggingFace tokenizer."""

    def __call__(self, text, truncation=True, padding="max_length", max_length=16, return_tensors="pt"):
        ids = [ord(c) % 50 + 1 for c in text[:max_length]]
        ids += [0] * (max_length - len(ids))
        mask = [1] * min(len(text), max_length) + [0] * (max_length - min(len(text), max_length))
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.tensor([mask]),
        }


def test_smiles_dataset_tokenizes_and_carries_labels():
    smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    labels = np.array([0, 1, 0])
    ds = SmilesDataset(smiles, labels, _FakeTokenizer(), max_length=16)
    assert len(ds) == 3

    item = ds[1]
    assert set(item.keys()) >= {"input_ids", "attention_mask", "labels"}
    assert item["input_ids"].shape[0] == 16
    assert item["attention_mask"].shape[0] == 16
    assert float(item["labels"]) == 1.0


def test_smiles_dataset_without_labels_has_no_label_key():
    ds = SmilesDataset(["CCO"], None, _FakeTokenizer(), max_length=8)
    item = ds[0]
    assert "labels" not in item
    assert item["input_ids"].shape[0] == 8
