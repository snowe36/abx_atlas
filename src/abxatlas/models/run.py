"""Run Gram-negative QSAR with random / scaffold / time leakage diagnostics."""

from __future__ import annotations

import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone

from abxatlas.config import RANDOM_STATE
from abxatlas.data.curate import load_curated
from abxatlas.featurize.fingerprints import morgan_fps
from abxatlas.models.interpret import run_interpretation
from abxatlas.models.qsar import SplitResult, evaluate_split, make_models
from abxatlas.models.splits import (
    random_split_indices,
    scaffold_split_indices,
    time_split_indices,
)
from abxatlas.paths import FIGURES, PROCESSED, ensure_dirs

logger = logging.getLogger(__name__)


def _prepare_gram_neg_task(compounds: pd.DataFrame) -> pd.DataFrame:
    df = compounds.dropna(subset=["smiles", "gram_neg_active"]).copy()
    df = df[df["has_gram_neg_assay"] == True]  # noqa: E712
    df["gram_neg_active"] = df["gram_neg_active"].astype(int)
    return df.reset_index(drop=True)


def run_qsar(test_size: float = 0.2) -> pd.DataFrame:
    ensure_dirs()
    _, compounds = load_curated()
    df = _prepare_gram_neg_task(compounds)
    if len(df) < 50:
        raise RuntimeError(
            f"Too few Gram-negative labeled compounds ({len(df)}). "
            "Re-run download with a higher max_per_organism."
        )

    X, mask = morgan_fps(df["smiles"].tolist())
    df = df.loc[mask].reset_index(drop=True)
    X = X[mask].astype(np.float64)
    y = df["gram_neg_active"].to_numpy()

    splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    splits["random"] = random_split_indices(len(df), test_size=test_size, y=y)
    splits["scaffold"] = scaffold_split_indices(
        df["smiles"], test_size=test_size, random_state=RANDOM_STATE
    )
    if "document_year_min" in df.columns:
        tsplit = time_split_indices(df["document_year_min"], test_fraction=test_size)
        if tsplit is not None:
            splits["time"] = tsplit

    all_results: list[SplitResult] = []
    models = make_models(RANDOM_STATE)
    for split_name, (tr, te) in splits.items():
        all_results.extend(
            evaluate_split(X, y, tr, te, split_name=split_name, models=models)
        )

    results_df = pd.DataFrame([r.__dict__ for r in all_results])
    out_csv = PROCESSED / "qsar_leakage_results.csv"
    results_df.to_csv(out_csv, index=False)

    _plot_leakage(results_df)
    gap = _optimistic_gap(results_df)

    interpret_meta = _run_scaffold_interpretation(df, X, y, splits["scaffold"])

    meta = {
        "n_compounds": int(len(df)),
        "active_rate": float(y.mean()),
        "results_csv": str(out_csv),
        "optimistic_gap_roc_auc": gap,
        "primary_task": "gram_neg_active (pChEMBL >= 5)",
        "interpretation": interpret_meta,
    }
    (PROCESSED / "qsar_meta.json").write_text(json.dumps(meta, indent=2))
    print(results_df.to_string(index=False))
    if gap:
        print(
            f"\nOptimistic gap (random − scaffold ROC-AUC, mean across models): "
            f"{gap.get('mean_gap_roc_auc', float('nan')):.3f}"
        )
    if interpret_meta and interpret_meta.get("error_counts"):
        ec = interpret_meta["error_counts"]
        print(
            f"Scaffold-split logreg errors — "
            f"TP={ec.get('TP', 0)} FP={ec.get('FP', 0)} "
            f"FN={ec.get('FN', 0)} TN={ec.get('TN', 0)}"
        )
    return results_df


def _run_scaffold_interpretation(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    scaffold_split: tuple[np.ndarray, np.ndarray],
) -> dict:
    """Train logreg on scaffold-split train; interpret on held-out scaffolds."""
    tr, te = scaffold_split
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        logger.warning("Skipping interpretation: need both classes in train and test")
        return {}
    model = clone(make_models(RANDOM_STATE)["logreg"])
    model.fit(X[tr], y[tr])
    try:
        return run_interpretation(
            model,
            X_train=X[tr],
            y_train=y[tr],
            X_test=X[te],
            y_test=y[te],
            smiles_test=df.loc[te, "smiles"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Interpretation failed: %s", exc)
        return {"error": str(exc)}


def _optimistic_gap(results: pd.DataFrame) -> dict:
    if results.empty:
        return {}
    piv = results.pivot_table(
        index="model_name", columns="split_name", values="roc_auc", aggfunc="mean"
    )
    if "random" not in piv.columns or "scaffold" not in piv.columns:
        return {}
    gaps = (piv["random"] - piv["scaffold"]).dropna()
    return {
        "per_model": gaps.to_dict(),
        "mean_gap_roc_auc": float(gaps.mean()),
    }


def _plot_leakage(results: pd.DataFrame) -> None:
    if results.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    split_order = [
        s for s in ["random", "scaffold", "time"] if s in results["split_name"].unique()
    ]
    models = sorted(results["model_name"].unique())
    x = np.arange(len(split_order))
    width = 0.25
    colors = {"logreg": "#1b4f72", "rf": "#b9770e"}
    for i, model in enumerate(models):
        subset = results[results["model_name"] == model].set_index("split_name")
        vals = [
            subset.loc[s, "roc_auc"] if s in subset.index else np.nan for s in split_order
        ]
        ax.bar(x + i * width, vals, width=width, label=model, color=colors.get(model))
    ax.set_xticks(x + width)
    ax.set_xticklabels(split_order)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Gram-negative QSAR: random vs scaffold (leakage) splits")
    ax.axhline(0.5, color="#999999", lw=0.8, ls="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    out = FIGURES / "qsar_leakage_rocauc.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    logger.info("Wrote %s", out)
