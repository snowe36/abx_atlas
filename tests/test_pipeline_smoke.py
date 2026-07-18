"""Hermetic smoke test: fixture assays → curate → splits → CPU QSAR."""

from pathlib import Path

import numpy as np
import pandas as pd

from abxatlas.data.curate import curate_activities
from abxatlas.featurize.fingerprints import morgan_fps
from abxatlas.models.qsar import evaluate_split, make_models
from abxatlas.models.splits import (
    random_split_indices,
    scaffold_split_indices,
    time_split_indices,
)

FIXTURE = Path(__file__).parent / "fixtures" / "raw_assays.csv"


def test_fixture_pipeline_smoke(tmp_path, monkeypatch):
    """CI-safe end-to-end path that never hits ChEMBL or local data/."""
    from abxatlas import paths
    from abxatlas.data import curate as curate_mod

    monkeypatch.setattr(paths, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(paths, "RAW", tmp_path / "raw")
    monkeypatch.setattr(paths, "FIGURES", tmp_path / "figures")
    monkeypatch.setattr(curate_mod, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(curate_mod, "RAW", tmp_path / "raw")
    monkeypatch.setattr(curate_mod, "CURATED_PATH", tmp_path / "processed" / "curated.parquet")
    monkeypatch.setattr(curate_mod, "COMPOUND_PATH", tmp_path / "processed" / "compounds.parquet")
    (tmp_path / "processed").mkdir(parents=True)
    (tmp_path / "raw").mkdir(parents=True)

    raw = pd.read_csv(FIXTURE)
    curated, compounds = curate_activities(raw=raw)
    assert len(curated) > 0
    assert len(compounds) > 0

    df = compounds.dropna(subset=["smiles", "gram_neg_active"]).copy()
    df = df[df["has_gram_neg_assay"] == True]  # noqa: E712
    df["gram_neg_active"] = df["gram_neg_active"].astype(int)
    df = df.reset_index(drop=True)
    assert len(df) >= 40
    assert df["gram_neg_active"].nunique() == 2

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
    assert set(models) >= {"logreg", "rf", "gbdt"}

    all_rows = []
    for split_name, (tr, te) in splits.items():
        all_rows.extend(
            evaluate_split(X, y, tr, te, split_name=split_name, models=models, random_state=0)
        )

    results = pd.DataFrame([r.__dict__ for r in all_rows])
    assert not results.empty
    assert set(results["model_name"]) >= {"logreg", "rf", "gbdt"}
    assert results["roc_auc"].between(0.0, 1.0).all()
    # At least one split should be above chance for the strongest model
    assert float(results["roc_auc"].max()) > 0.55
