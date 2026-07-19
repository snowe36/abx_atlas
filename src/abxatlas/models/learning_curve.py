"""Scaffold-split learning curves: does more data fix generalization?"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from abxatlas.config import RANDOM_STATE
from abxatlas.models.qsar import make_models
from abxatlas.paths import FIGURES, PROCESSED, ensure_dirs

logger = logging.getLogger(__name__)

DEFAULT_FRACTIONS = (0.1, 0.2, 0.4, 0.6, 0.8, 1.0)


def scaffold_learning_curve(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Evaluate models on a fixed scaffold test set while growing train size."""
    rng = np.random.RandomState(random_state)
    tr = np.array(train_idx)
    te = np.array(test_idx)
    X_test, y_test = X[te], y[te]
    if len(np.unique(y_test)) < 2:
        return pd.DataFrame()

    models = {
        k: v for k, v in make_models(random_state).items() if k in ("logreg", "rf")
    }
    rows = []
    for frac in fractions:
        n = max(20, int(round(len(tr) * frac)))
        n = min(n, len(tr))
        # Stratified subsample of train
        chosen = _stratified_subsample(tr, y[tr], n, rng)
        if len(np.unique(y[chosen])) < 2:
            continue
        for name, model in models.items():
            clf = clone(model)
            clf.fit(X[chosen], y[chosen])
            proba = clf.predict_proba(X_test)[:, 1]
            rows.append(
                {
                    "train_fraction": float(frac),
                    "n_train": int(len(chosen)),
                    "model_name": name,
                    "roc_auc": float(roc_auc_score(y_test, proba)),
                }
            )
    return pd.DataFrame(rows)


def _stratified_subsample(
    idx: np.ndarray, y: np.ndarray, n: int, rng: np.random.RandomState
) -> np.ndarray:
    if n >= len(idx):
        return idx
    pos = idx[y == 1]
    neg = idx[y == 0]
    n_pos = max(1, int(round(n * (len(pos) / len(idx)))))
    n_pos = min(n_pos, len(pos))
    n_neg = min(n - n_pos, len(neg))
    if n_neg < 1 and len(neg):
        n_neg = 1
        n_pos = min(n - 1, len(pos))
    take_pos = rng.choice(pos, size=n_pos, replace=False) if len(pos) else np.array([], dtype=int)
    take_neg = rng.choice(neg, size=n_neg, replace=False) if len(neg) else np.array([], dtype=int)
    out = np.concatenate([take_pos, take_neg])
    rng.shuffle(out)
    return out


def plot_learning_curve(
    curve: pd.DataFrame,
    out_path: Path | None = None,
) -> Path | None:
    ensure_dirs()
    if curve.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = {"logreg": "#1b4f72", "rf": "#b9770e"}
    for model, g in curve.groupby("model_name"):
        g = g.sort_values("n_train")
        ax.plot(
            g["n_train"],
            g["roc_auc"],
            marker="o",
            label=model,
            color=colors.get(model, None),
            lw=2,
        )
    ax.axhline(0.5, color="#999999", lw=0.8, ls="--")
    ax.set_xlabel("Training compounds (scaffold-split train subsample)")
    ax.set_ylabel("ROC-AUC on held-out scaffolds")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Figure 4. Does more ChEMBL data fix scaffold generalization?")
    ax.legend(frameon=False)
    fig.tight_layout()
    out = out_path or (FIGURES / "fig4_learning_curve.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    logger.info("Wrote %s", out)
    return out


def run_learning_curve(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict:
    curve = scaffold_learning_curve(X, y, train_idx, test_idx)
    path = PROCESSED / "qsar_learning_curve.csv"
    curve.to_csv(path, index=False)
    fig = plot_learning_curve(curve)
    return {
        "csv": str(path),
        "figure": str(fig) if fig else None,
        "plateau_hint": _plateau_hint(curve),
    }


def _plateau_hint(curve: pd.DataFrame) -> str:
    if curve.empty:
        return "n/a"
    hints = []
    for model, g in curve.groupby("model_name"):
        g = g.sort_values("n_train")
        if len(g) < 2:
            continue
        delta = float(g["roc_auc"].iloc[-1] - g["roc_auc"].iloc[0])
        last_step = float(g["roc_auc"].iloc[-1] - g["roc_auc"].iloc[-2])
        hints.append(f"{model}: Δ={delta:+.3f} overall, last step {last_step:+.3f}")
    return "; ".join(hints)
