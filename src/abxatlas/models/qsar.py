"""Sklearn QSAR classifiers on Morgan fingerprints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class SplitResult:
    split_name: str
    model_name: str
    n_train: int
    n_test: int
    roc_auc: float
    average_precision: float
    balanced_accuracy: float


def make_models(random_state: int = 42) -> dict[str, object]:
    # logreg + RF + GBDT on Morgan bits
    return {
        "logreg": Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "rf": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "gbdt": HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.08,
            max_iter=250,
            min_samples_leaf=10,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=random_state,
        ),
    }


def evaluate_split(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    split_name: str,
    models: dict[str, object] | None = None,
    random_state: int = 42,
) -> list[SplitResult]:
    models = models or make_models(random_state)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    results = []
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return results

    for name, model in models.items():
        clf = clone(model)
        clf.fit(X_train, y_train)
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_test)[:, 1]
        else:
            proba = clf.decision_function(X_test)
            proba = (proba - proba.min()) / (proba.max() - proba.min() + 1e-9)
        pred = (proba >= 0.5).astype(int)
        results.append(
            SplitResult(
                split_name=split_name,
                model_name=name,
                n_train=int(len(train_idx)),
                n_test=int(len(test_idx)),
                roc_auc=float(roc_auc_score(y_test, proba)),
                average_precision=float(average_precision_score(y_test, proba)),
                balanced_accuracy=float(balanced_accuracy_score(y_test, pred)),
            )
        )
    return results
