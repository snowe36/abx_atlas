"""Optuna hyperparameter sweeps for the GPU deep models.

Both sweeps only ever see the *outer* split's train set — the inner
validation carve-out used for scoring trials never touches the held-out
test scaffolds / random rows / future-year rows used for final reporting.
Imports optuna at module scope (part of the optional `gpu` extra); only
ever imported lazily by run.py when an HPO sweep is requested.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import optuna
import pandas as pd

from abxatlas.config import RANDOM_STATE
from abxatlas.paths import PROCESSED, ensure_dirs

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _inner_train_val_split(
    train_idx: np.ndarray, inner_val_fraction: float, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(random_state)
    idx = np.array(train_idx)
    shuffled = rng.permutation(idx)
    n_val = max(20, int(round(len(shuffled) * inner_val_fraction)))
    n_val = min(n_val, len(shuffled) - 20) if len(shuffled) > 40 else len(shuffled) // 2
    return shuffled[n_val:], shuffled[:n_val]  # inner_train, inner_val


def run_gnn_hpo(
    graphs: list,
    y: np.ndarray,
    train_idx: np.ndarray,
    n_trials: int = 20,
    epochs: int = 30,
    inner_val_fraction: float = 0.2,
    random_state: int = RANDOM_STATE,
    trials_csv: str = "gnn_hpo_trials.csv",
) -> dict[str, Any]:
    """Search GNN architecture/optimizer hyperparameters, scored on an inner
    train/val split carved out of `train_idx`. Returns the best config and
    writes every trial to `data/processed/{trials_csv}`."""
    from abxatlas.models.gnn import evaluate_gnn_split

    ensure_dirs()
    inner_train, inner_val = _inner_train_val_split(train_idx, inner_val_fraction, random_state)
    trial_rows: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        config = {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
            "num_layers": trial.suggest_int("num_layers", 2, 4),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "conv_type": trial.suggest_categorical("conv_type", ["gcn", "gin"]),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        }
        results = evaluate_gnn_split(
            graphs,
            y,
            inner_train,
            inner_val,
            split_name="hpo_inner",
            model_name="gnn_hpo",
            config=config,
            epochs=epochs,
            random_state=random_state,
        )
        auc = results[0].roc_auc if results else 0.0
        trial_rows.append({**config, "trial": trial.number, "val_roc_auc": auc})
        return auc

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    out_csv = PROCESSED / trials_csv
    pd.DataFrame(trial_rows).to_csv(out_csv, index=False)
    logger.info("GNN HPO best val ROC-AUC=%.4f params=%s", study.best_value, study.best_params)
    return {
        "best_params": study.best_params,
        "best_value": float(study.best_value),
        "n_trials": n_trials,
        "trials_csv": str(out_csv),
    }


def run_pretrained_hpo(
    smiles: Sequence[str],
    y: np.ndarray,
    train_idx: np.ndarray,
    model_name_hf: str | None = None,
    n_trials: int = 6,
    inner_val_fraction: float = 0.2,
    random_state: int = RANDOM_STATE,
    trials_csv: str = "pretrained_hpo_trials.csv",
) -> dict[str, Any]:
    """Lighter sweep for the pretrained transformer (fewer trials — each
    fine-tune epoch is much costlier than a GNN epoch)."""
    from abxatlas.models.pretrained import DEFAULT_MODEL_NAME, evaluate_pretrained_split

    ensure_dirs()
    model_name_hf = model_name_hf or DEFAULT_MODEL_NAME
    inner_train, inner_val = _inner_train_val_split(train_idx, inner_val_fraction, random_state)
    trial_rows: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        config = {
            "lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
            "epochs": trial.suggest_int("epochs", 2, 5),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        }
        results = evaluate_pretrained_split(
            smiles,
            y,
            inner_train,
            inner_val,
            split_name="hpo_inner",
            model_name="chemberta_hpo",
            model_name_hf=model_name_hf,
            config=config,
            random_state=random_state,
        )
        auc = results[0].roc_auc if results else 0.0
        trial_rows.append({**config, "trial": trial.number, "val_roc_auc": auc})
        return auc

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    out_csv = PROCESSED / trials_csv
    pd.DataFrame(trial_rows).to_csv(out_csv, index=False)
    logger.info(
        "Pretrained HPO best val ROC-AUC=%.4f params=%s", study.best_value, study.best_params
    )
    return {
        "best_params": study.best_params,
        "best_value": float(study.best_value),
        "n_trials": n_trials,
        "trials_csv": str(out_csv),
    }
