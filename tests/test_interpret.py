"""Unit tests for QSAR interpretation helpers."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from abxatlas.models.interpret import (
    error_labels,
    extract_logreg_weights,
    scaffold_error_enrichment,
)


def _toy_logreg():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(80, 32)).astype(float)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    y[0] = 1 - y[0]  # ensure both classes
    pipe = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=500, random_state=0)),
        ]
    )
    pipe.fit(X, y)
    return pipe, X, y


def test_error_labels():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    assert error_labels(y_true, y_pred).tolist() == ["TP", "FN", "FP", "TN"]


def test_extract_logreg_weights():
    pipe, _, _ = _toy_logreg()
    weights = extract_logreg_weights(pipe)
    assert len(weights) == 32
    assert {"bit", "weight", "abs_weight"} <= set(weights.columns)
    assert weights["abs_weight"].is_monotonic_decreasing


def test_scaffold_error_enrichment():
    # Two benzene, two pyridine scaffolds
    smiles = ["c1ccccc1", "Cc1ccccc1", "c1ccncc1", "Cc1ccncc1"]
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 0])  # FN, FP, TP, TN
    enrich = scaffold_error_enrichment(smiles, y_true, y_pred, min_count=1)
    assert not enrich.empty
    assert set(enrich["error_rate"]) <= {0.0, 0.5, 1.0}
