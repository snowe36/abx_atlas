"""Self-contained demo: fixture assays → leakage story (no ChEMBL download)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from abxatlas.paths import FIGURES, PROCESSED, ROOT, ensure_dirs

DEMO = ROOT / "demo"
DEMO_OUT = DEMO / "outputs"
DEMO_SCRATCH = DEMO / "scratch"
FIXTURE = ROOT / "tests" / "fixtures" / "raw_assays.csv"

_HERO_FIGS = [
    "fig3_leakage_rocauc.png",
    "fig7_chemotype_temporal_failures.png",
    "fig8_surprise_case_study.png",
    "fig1_chemspace_atlas.png",
    "fig4_learning_curve.png",
]


def _fixture_leakage(figures_dir: Path) -> Path:
    """Hermetic fixture QSAR into demo/scratch — does not touch data/processed/."""
    from abxatlas import paths
    from abxatlas.data import curate as curate_mod
    from abxatlas.data.curate import curate_activities
    from abxatlas.featurize.fingerprints import morgan_fps
    from abxatlas.models.qsar import evaluate_split, make_models
    from abxatlas.models.run import _plot_leakage
    from abxatlas.models.splits import (
        random_split_indices,
        scaffold_split_indices,
        time_split_indices,
    )

    scratch = DEMO_SCRATCH
    scratch.mkdir(parents=True, exist_ok=True)
    fig_dir = figures_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Redirect writes away from the full-atlas processed tables.
    paths.PROCESSED = scratch
    paths.RAW = scratch / "raw"
    paths.FIGURES = fig_dir
    paths.RAW.mkdir(parents=True, exist_ok=True)
    curate_mod.PROCESSED = scratch
    curate_mod.RAW = paths.RAW
    curate_mod.CURATED_PATH = scratch / "antibacterial_curated.parquet"
    curate_mod.COMPOUND_PATH = scratch / "compounds.parquet"

    raw = pd.read_csv(FIXTURE)
    _curated, compounds = curate_activities(raw=raw)
    df = compounds.dropna(subset=["smiles", "gram_neg_active"]).copy()
    df = df[df["has_gram_neg_assay"] == True]  # noqa: E712
    df["gram_neg_active"] = df["gram_neg_active"].astype(int)
    df = df.reset_index(drop=True)

    X, mask = morgan_fps(df["smiles"].tolist())
    df = df.loc[mask].reset_index(drop=True)
    X = X[mask].astype(np.float64)
    y = df["gram_neg_active"].to_numpy()

    splits = {
        "random": random_split_indices(len(df), test_size=0.3, y=y),
        "scaffold": scaffold_split_indices(df["smiles"], test_size=0.3, random_state=0),
    }
    tsplit = time_split_indices(df["document_year_min"], test_fraction=0.3)
    if tsplit is not None:
        splits["time"] = tsplit

    models = make_models(random_state=0)
    rows = []
    for split_name, (tr, te) in splits.items():
        rows.extend(
            evaluate_split(
                X, y, tr, te, split_name=split_name, models=models, random_state=0
            )
        )
    results = pd.DataFrame([r.__dict__ for r in rows])
    out = DEMO_OUT / "fixture_leakage_results.csv"
    results.to_csv(out, index=False)

    # _plot_leakage writes into paths.FIGURES
    import abxatlas.models.run as run_mod

    run_mod.FIGURES = fig_dir
    _plot_leakage(results)
    return out


def run_demo() -> dict:
    """Fixture leakage run + stage committed full-atlas hero figures."""
    import os

    os.environ.setdefault("MPLBACKEND", "Agg")
    ensure_dirs()
    DEMO_OUT.mkdir(parents=True, exist_ok=True)

    print("=== abx_atlas demo (fixture assays — no ChEMBL download) ===")
    fixture_fig_dir = DEMO_OUT / "_fixture_figures"
    demo_csv = _fixture_leakage(fixture_fig_dir)
    fixture_png = fixture_fig_dir / "fig3_leakage_rocauc.png"
    if fixture_png.exists():
        shutil.copy2(fixture_png, DEMO_OUT / "fixture_leakage_rocauc.png")

    copied = [demo_csv.name]
    if (DEMO_OUT / "fixture_leakage_rocauc.png").exists():
        copied.append("fixture_leakage_rocauc.png")

    # Replot full-atlas hero from committed metrics (palette refresh) when present.
    leakage = PROCESSED / "qsar_leakage_results.csv"
    if leakage.exists():
        from abxatlas.models import run as run_mod

        run_mod.FIGURES = FIGURES
        run_mod._plot_leakage(pd.read_csv(leakage))

    for name in _HERO_FIGS:
        src = FIGURES / name
        if src.exists():
            shutil.copy2(src, DEMO_OUT / name)
            copied.append(name)

    summary = {
        "outputs": str(DEMO_OUT),
        "copied": copied,
        "headline": (
            "Fixture run shows the split gap in miniature; "
            "fig3_leakage_rocauc.png is the full ChEMBL hero when present."
        ),
    }
    print("")
    print(f"Demo outputs → {DEMO_OUT}")
    for c in sorted(set(copied)):
        print(f"  - {c}")
    print(summary["headline"])
    return summary
