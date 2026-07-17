"""Random, scaffold, and time splits for leakage diagnostics."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from abxatlas.config import RANDOM_STATE
from abxatlas.featurize.scaffolds import scaffold_series


def random_split_indices(
    n: int,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    y: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    strat = y if y is not None and len(np.unique(y)) > 1 else None
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=random_state, stratify=strat
    )
    return train_idx, test_idx


def scaffold_split_indices(
    smiles: list[str] | pd.Series,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Group Bemis–Murcko scaffolds into train/test without scaffold leakage.

    Greedy assignment of largest scaffolds first toward the target test fraction.
    """
    scaffolds = scaffold_series(list(smiles))
    # Map scaffold -> indices
    scaf_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, scaf in enumerate(scaffolds):
        key = scaf if scaf is not None else f"__none_{i}"
        scaf_to_idx[key].append(i)

    rng = np.random.RandomState(random_state)
    items = sorted(scaf_to_idx.items(), key=lambda kv: len(kv[1]), reverse=True)
    # Shuffle equal-size buckets slightly for stability
    rng.shuffle(items)

    n = len(smiles)
    target_test = int(round(n * test_size))
    test_idx: list[int] = []
    train_idx: list[int] = []
    for _, idxs in items:
        if len(test_idx) < target_test:
            test_idx.extend(idxs)
        else:
            train_idx.extend(idxs)
    # Ensure non-empty train
    if not train_idx and test_idx:
        move = test_idx.pop()
        train_idx.append(move)
    return np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)


def time_split_indices(
    years: pd.Series | np.ndarray,
    test_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Hold out the most recent fraction of compounds by earliest document year."""
    y = pd.to_numeric(pd.Series(years), errors="coerce")
    if y.isna().all():
        return None
    order = np.argsort(y.fillna(y.min() - 1).to_numpy())
    n = len(order)
    n_test = max(1, int(round(n * test_fraction)))
    test_idx = order[-n_test:]
    train_idx = order[:-n_test]
    if len(train_idx) == 0:
        return None
    return train_idx.astype(int), test_idx.astype(int)
