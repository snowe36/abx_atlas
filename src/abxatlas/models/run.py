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
from abxatlas.featurize.graphs import smiles_to_graphs
from abxatlas.models.interpret import run_interpretation
from abxatlas.models.learning_curve import run_learning_curve
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


def run_qsar(
    test_size: float = 0.2,
    with_gnn: bool = False,
    gnn_epochs: int = 60,
    gnn_hpo_trials: int = 0,
    with_pretrained: bool = False,
    pretrained_model: str = "seyonec/ChemBERTa-zinc-base-v1",
    pretrained_epochs: int = 3,
    pretrained_hpo_trials: int = 0,
) -> pd.DataFrame:
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

    deep_meta: dict = {}
    if with_gnn:
        try:
            deep_meta["gnn"] = _run_gnn_models(
                df,
                splits,
                gnn_epochs=gnn_epochs,
                gnn_hpo_trials=gnn_hpo_trials,
                all_results=all_results,
            )
        except Exception as exc:  # noqa: BLE001 — never let a GPU model crash the benchmark
            logger.warning("--with-gnn failed, continuing without it: %s", exc)
            deep_meta["gnn"] = {"error": str(exc)}
    if with_pretrained:
        try:
            deep_meta["pretrained"] = _run_pretrained_models(
                df,
                splits,
                pretrained_model=pretrained_model,
                pretrained_epochs=pretrained_epochs,
                pretrained_hpo_trials=pretrained_hpo_trials,
                all_results=all_results,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("--with-pretrained failed, continuing without it: %s", exc)
            deep_meta["pretrained"] = {"error": str(exc)}

    results_df = pd.DataFrame([r.__dict__ for r in all_results])
    out_csv = PROCESSED / "qsar_leakage_results.csv"
    results_df.to_csv(out_csv, index=False)

    _plot_leakage(results_df)
    gap = _optimistic_gap(results_df)

    interpret_meta = _run_scaffold_interpretation(df, X, y, splits["scaffold"])
    curve_meta = {}
    try:
        curve_meta = run_learning_curve(X, y, splits["scaffold"][0], splits["scaffold"][1])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Learning curve failed: %s", exc)

    meta = {
        "n_compounds": int(len(df)),
        "active_rate": float(y.mean()),
        "results_csv": str(out_csv),
        "optimistic_gap_roc_auc": gap,
        "primary_task": "gram_neg_active (pChEMBL >= 5)",
        "interpretation": interpret_meta,
        "learning_curve": curve_meta,
        "deep_models": deep_meta,
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
    if curve_meta.get("plateau_hint"):
        print(f"Learning curve: {curve_meta['plateau_hint']}")
    return results_df


def _run_gnn_models(
    df: pd.DataFrame,
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    gnn_epochs: int,
    gnn_hpo_trials: int,
    all_results: list[SplitResult],
) -> dict:
    """Featurize once, optionally HPO-tune, then evaluate the GNN on every
    outer split — appending SplitResult rows into `all_results` in place."""
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
    except ImportError as exc:
        logger.warning(
            "Skipping --with-gnn: torch/torch_geometric not installed (%s). "
            "Install with: pip install -e '.[gpu]'",
            exc,
        )
        return {"skipped": str(exc)}

    from abxatlas.models.gnn import DEFAULT_GNN_CONFIG, evaluate_gnn_split

    graphs, gmask = smiles_to_graphs(df["smiles"].tolist())
    if not np.all(gmask):
        logger.warning(
            "GNN: dropped %d / %d molecules that failed graph featurization",
            int((~gmask).sum()),
            len(gmask),
        )
    y_all = df["gram_neg_active"].to_numpy().astype(int)
    y_graphs = y_all[gmask]
    valid_outer = np.flatnonzero(gmask)
    outer_to_graph = {int(outer): gi for gi, outer in enumerate(valid_outer)}

    def _to_graph_idx(outer_idx: np.ndarray) -> np.ndarray:
        return np.array(
            [outer_to_graph[int(i)] for i in outer_idx if int(i) in outer_to_graph], dtype=int
        )

    config = dict(DEFAULT_GNN_CONFIG)
    hpo_meta: dict = {}
    if gnn_hpo_trials > 0 and "scaffold" in splits:
        from abxatlas.models.hpo import run_gnn_hpo

        tr_scaffold, _ = splits["scaffold"]
        tr_graph_idx = _to_graph_idx(tr_scaffold)
        try:
            hpo_meta = run_gnn_hpo(
                graphs,
                y_graphs,
                tr_graph_idx,
                n_trials=gnn_hpo_trials,
                epochs=min(gnn_epochs, 30),
            )
            config.update(hpo_meta.get("best_params", {}))
        except Exception as exc:  # noqa: BLE001
            logger.warning("GNN HPO failed, falling back to defaults: %s", exc)
            hpo_meta = {"error": str(exc)}

    for split_name, (tr, te) in splits.items():
        tr_g, te_g = _to_graph_idx(tr), _to_graph_idx(te)
        if len(tr_g) == 0 or len(te_g) == 0:
            continue
        all_results.extend(
            evaluate_gnn_split(
                graphs,
                y_graphs,
                tr_g,
                te_g,
                split_name=split_name,
                config=config,
                epochs=gnn_epochs,
            )
        )
    return {"config": config, "hpo": hpo_meta, "n_graphs": int(len(graphs))}


def _run_pretrained_models(
    df: pd.DataFrame,
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    pretrained_model: str,
    pretrained_epochs: int,
    pretrained_hpo_trials: int,
    all_results: list[SplitResult],
) -> dict:
    """Fine-tune (with optional HPO) and evaluate the pretrained transformer
    on every outer split — appending SplitResult rows into `all_results`."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        logger.warning(
            "Skipping --with-pretrained: torch/transformers not installed (%s). "
            "Install with: pip install -e '.[gpu]'",
            exc,
        )
        return {"skipped": str(exc)}

    from abxatlas.models.pretrained import DEFAULT_PRETRAINED_CONFIG, evaluate_pretrained_split

    smiles = df["smiles"].tolist()
    y_all = df["gram_neg_active"].to_numpy().astype(int)

    config = {**DEFAULT_PRETRAINED_CONFIG, "epochs": pretrained_epochs}
    hpo_meta: dict = {}
    if pretrained_hpo_trials > 0 and "scaffold" in splits:
        from abxatlas.models.hpo import run_pretrained_hpo

        tr_scaffold, _ = splits["scaffold"]
        try:
            hpo_meta = run_pretrained_hpo(
                smiles,
                y_all,
                tr_scaffold,
                model_name_hf=pretrained_model,
                n_trials=pretrained_hpo_trials,
            )
            config.update(hpo_meta.get("best_params", {}))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Pretrained HPO failed, falling back to defaults: %s", exc)
            hpo_meta = {"error": str(exc)}

    for split_name, (tr, te) in splits.items():
        all_results.extend(
            evaluate_pretrained_split(
                smiles,
                y_all,
                tr,
                te,
                split_name=split_name,
                model_name_hf=pretrained_model,
                config=config,
            )
        )
    return {"config": config, "hpo": hpo_meta, "model_name_hf": pretrained_model}


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
            smiles_train=df.loc[tr, "smiles"],
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
    """Figure 3 (hero): random vs scaffold vs time ROC-AUC."""
    if results.empty:
        return
    split_order = [
        s for s in ["random", "scaffold", "time"] if s in results["split_name"].unique()
    ]
    models = sorted(results["model_name"].unique())
    fig, ax = plt.subplots(figsize=(7.5 + 0.6 * max(0, len(models) - 2), 4.8))
    x = np.arange(len(split_order))
    width = 0.8 / max(len(models), 1)
    base_colors = {"logreg": "#1b4f72", "rf": "#b9770e", "gnn": "#1e8449", "chemberta": "#6c3483"}
    palette = plt.get_cmap("tab10")
    colors = {m: base_colors.get(m, palette(i % 10)) for i, m in enumerate(models)}
    for i, model in enumerate(models):
        subset = results[results["model_name"] == model].set_index("split_name")
        vals = [
            subset.loc[s, "roc_auc"] if s in subset.index else np.nan for s in split_order
        ]
        bars = ax.bar(
            x + i * width, vals, width=width, label=model, color=colors.get(model)
        )
        for b, v in zip(bars, vals, strict=True):
            if np.isfinite(v):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    v + 0.015,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(split_order)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.4, 1.05)
    ax.set_title("Figure 3. Leakage-aware Gram-negative QSAR (hero result)")
    ax.axhline(0.5, color="#999999", lw=0.8, ls="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    for name in ("qsar_leakage_rocauc.png", "fig3_leakage_rocauc.png"):
        out = FIGURES / name
        fig.savefig(out, dpi=200)
        logger.info("Wrote %s", out)
    plt.close(fig)
