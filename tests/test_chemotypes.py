"""Unit tests for named chemotype assignment and enrichment."""

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from abxatlas.models.chemotypes import (
    assign_chemotype_families,
    chemotype_error_enrichment,
    load_chemotype_families,
    surprise_case_study,
)


def test_load_chemotype_families():
    fams = load_chemotype_families()
    ids = {f["id"] for f in fams}
    assert {"fluoroquinolone", "glycopeptide", "polymyxin", "beta_lactam"} <= ids
    assert any(f.get("case_study") for f in fams)


def test_assign_fluoroquinolone():
    # Ciprofloxacin
    cipro = "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O"
    benzene = "c1ccccc1"
    out = assign_chemotype_families([cipro, benzene])
    assert out.loc[0, "primary_family"] == "fluoroquinolone"
    assert out.loc[1, "primary_family"] == "other"


def test_assign_beta_lactam():
    amox = "CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O"
    out = assign_chemotype_families([amox])
    assert out.loc[0, "primary_family"] == "beta_lactam"


def test_chemotype_error_enrichment():
    # Two FQs wrong, two benzene correct
    smiles = [
        "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
        "O=C(O)c1cn(C2CC2)c2ccc(F)cc2c1=O",
        "c1ccccc1",
        "Cc1ccccc1",
        "c1ccncc1",
        "Cc1ccncc1",
    ]
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([0, 0, 0, 0, 1, 0])  # both FQs are FN
    enrich = chemotype_error_enrichment(
        smiles, y_true, y_pred, split_name="time", min_count=2
    )
    assert not enrich.empty
    fq = enrich[enrich["family"] == "fluoroquinolone"]
    assert len(fq) == 1
    assert float(fq["error_rate"].iloc[0]) == 1.0
    assert float(fq["error_lift"].iloc[0]) > 1.0


def test_surprise_case_study_smoke():
    rng = np.random.default_rng(0)
    smiles = [
        "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
        "O=C(O)c1cn(CC)c2ccc(F)cc2c1=O",
        "c1ccccc1",
        "Cc1ccccc1",
        "c1ccncc1",
        "Cc1ccncc1",
        "c1ccc(O)cc1",
        "CCN(CC)c1ccccc1",
    ]
    n = len(smiles)
    X = rng.integers(0, 2, size=(n, 32)).astype(float)
    y = np.array([1, 1, 0, 0, 1, 0, 0, 1])
    tr = np.array([2, 3, 4, 5, 6, 7])
    te = np.array([0, 1])
    pipe = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=500, random_state=0)),
        ]
    )
    model = clone(pipe)
    model.fit(X[tr], y[tr])
    case = surprise_case_study(
        smiles,
        y,
        X,
        {"scaffold": (tr, te)},
        models_by_split={"scaffold": model},
        case_family_ids=("fluoroquinolone", "glycopeptide", "polymyxin"),
    )
    assert not case.empty
    fq = case[case["family"] == "fluoroquinolone"]
    assert int(fq["n_total"].iloc[0]) >= 2
    assert int(fq["n_test"].iloc[0]) == 2
