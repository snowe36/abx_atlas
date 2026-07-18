"""Model interpretation: logreg bit weights and FP/FN scaffold enrichment."""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from sklearn.pipeline import Pipeline

from abxatlas.config import MORGAN_NBITS, MORGAN_RADIUS
from abxatlas.featurize.scaffolds import scaffold_series
from abxatlas.paths import FIGURES, PROCESSED, ensure_dirs

logger = logging.getLogger(__name__)

def extract_logreg_weights(model: Pipeline) -> pd.DataFrame:
    """Return Morgan-bit coefficients from a fitted logreg pipeline."""
    clf = model.named_steps["clf"]
    coef = np.asarray(clf.coef_).ravel()
    return (
        pd.DataFrame({"bit": np.arange(len(coef)), "weight": coef})
        .assign(abs_weight=lambda d: d["weight"].abs())
        .sort_values("abs_weight", ascending=False)
        .reset_index(drop=True)
    )


def predict_binary(model, X: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Return (proba, pred) for a fitted classifier."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        raw = model.decision_function(X)
        proba = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    pred = (proba >= threshold).astype(int)
    return proba, pred


def error_labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Map predictions to TP / FP / FN / TN strings."""
    out = np.empty(len(y_true), dtype=object)
    yt = y_true.astype(int)
    yp = y_pred.astype(int)
    out[(yt == 1) & (yp == 1)] = "TP"
    out[(yt == 0) & (yp == 1)] = "FP"
    out[(yt == 1) & (yp == 0)] = "FN"
    out[(yt == 0) & (yp == 0)] = "TN"
    return out


def scaffold_error_enrichment(
    smiles: list[str] | pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_count: int = 2,
) -> pd.DataFrame:
    """Per-scaffold error counts on a held-out set (FP/FN focus)."""
    scaf = scaffold_series(list(smiles)).fillna("unknown").astype(str)
    err = error_labels(y_true, y_pred)
    frame = pd.DataFrame(
        {
            "scaffold": scaf.to_numpy(),
            "y_true": y_true.astype(int),
            "y_pred": y_pred.astype(int),
            "error": err,
        }
    )
    rows = []
    for scaffold, g in frame.groupby("scaffold"):
        n = len(g)
        if n < min_count:
            continue
        counts = g["error"].value_counts()
        fp = int(counts.get("FP", 0))
        fn = int(counts.get("FN", 0))
        tp = int(counts.get("TP", 0))
        tn = int(counts.get("TN", 0))
        rows.append(
            {
                "scaffold": scaffold,
                "n": n,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "fp_rate": fp / n,
                "fn_rate": fn / n,
                "error_rate": (fp + fn) / n,
                "active_rate": float(g["y_true"].mean()),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "scaffold",
                "n",
                "TP",
                "FP",
                "FN",
                "TN",
                "fp_rate",
                "fn_rate",
                "error_rate",
                "active_rate",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["error_rate", "n"], ascending=[False, False]
    ).reset_index(drop=True)


def top_bit_permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    bit_indices: np.ndarray,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Column-shuffle importance for selected bits on an already-fitted model."""
    from sklearn.metrics import roc_auc_score

    if len(bit_indices) == 0 or len(np.unique(y)) < 2:
        return pd.DataFrame(columns=["bit", "importance_mean", "importance_std"])

    rng = np.random.default_rng(random_state)
    base_proba = model.predict_proba(X)[:, 1]
    baseline = float(roc_auc_score(y, base_proba))
    rows = []
    for bit in bit_indices:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, bit] = rng.permutation(Xp[:, bit])
            proba = model.predict_proba(Xp)[:, 1]
            drops.append(baseline - float(roc_auc_score(y, proba)))
        rows.append(
            {
                "bit": int(bit),
                "importance_mean": float(np.mean(drops)),
                "importance_std": float(np.std(drops)),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def plot_logreg_weights(
    weights: pd.DataFrame,
    top_n: int = 20,
    out_path: Path | None = None,
) -> Path:
    ensure_dirs()
    pos = weights.nlargest(top_n, "weight")
    neg = weights.nsmallest(top_n, "weight")
    panel = pd.concat([neg, pos]).sort_values("weight")

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#922b21" if w < 0 else "#1b4f72" for w in panel["weight"]]
    ax.barh(
        [f"bit {b}" for b in panel["bit"]],
        panel["weight"],
        color=colors,
    )
    ax.axvline(0, color="#666666", lw=0.8)
    ax.set_xlabel("Logistic regression coefficient")
    ax.set_title(f"Top ±{top_n} Morgan-bit weights (Gram-negative logreg)")
    fig.tight_layout()
    out = out_path or (FIGURES / "qsar_logreg_bit_weights.png")
    fig.savefig(out, dpi=180)
    plt.close(fig)
    logger.info("Wrote %s", out)
    return out


def _fragment_smiles_for_bit(
    mol,
    atom_idx: int,
    radius: int,
) -> str | None:
    """Return a SMILES for the Morgan environment centered on atom_idx."""
    try:
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, int(atom_idx))
        atoms = {int(atom_idx)}
        for bond_idx in env:
            bond = mol.GetBondWithIdx(bond_idx)
            atoms.add(bond.GetBeginAtomIdx())
            atoms.add(bond.GetEndAtomIdx())
        frag = Chem.PathToSubmol(mol, env) if env else None
        if frag is None:
            # radius-0: single atom
            return mol.GetAtomWithIdx(int(atom_idx)).GetSymbol()
        return Chem.MolToSmiles(frag, canonical=True)
    except Exception:  # noqa: BLE001
        return None


def explain_morgan_bits(
    smiles: list[str] | pd.Series,
    bit_indices: list[int] | np.ndarray,
    weights: pd.DataFrame | None = None,
    radius: int = MORGAN_RADIUS,
    n_bits: int = MORGAN_NBITS,
    examples_per_bit: int = 1,
) -> pd.DataFrame:
    """Map Morgan bit IDs to example substructure fragments from molecules that set them."""
    weight_map = {}
    if weights is not None and not weights.empty:
        weight_map = dict(zip(weights["bit"].astype(int), weights["weight"], strict=False))

    rows = []
    smiles_list = list(smiles)
    for bit in [int(b) for b in bit_indices]:
        found = 0
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
            if mol is None:
                continue
            bit_info: dict = {}
            AllChem.GetMorganFingerprintAsBitVect(
                mol, radius, nBits=n_bits, bitInfo=bit_info
            )
            if bit not in bit_info:
                continue
            atom_idx, env_radius = bit_info[bit][0]
            frag = _fragment_smiles_for_bit(mol, atom_idx, env_radius)
            rows.append(
                {
                    "bit": bit,
                    "weight": float(weight_map.get(bit, np.nan)),
                    "example_smiles": smi,
                    "center_atom": int(atom_idx),
                    "env_radius": int(env_radius),
                    "fragment_smiles": frag,
                }
            )
            found += 1
            if found >= examples_per_bit:
                break
        if found == 0:
            rows.append(
                {
                    "bit": bit,
                    "weight": float(weight_map.get(bit, np.nan)),
                    "example_smiles": None,
                    "center_atom": None,
                    "env_radius": None,
                    "fragment_smiles": None,
                }
            )
    return pd.DataFrame(rows)


def plot_morgan_bit_highlights(
    explanations: pd.DataFrame,
    out_path: Path | None = None,
    max_bits: int = 8,
) -> Path | None:
    """Draw RDKit Morgan-bit highlights for the strongest explained bits."""
    ensure_dirs()
    if explanations.empty:
        return None
    panel = explanations.dropna(subset=["example_smiles"]).copy()
    if panel.empty:
        return None
    if panel["weight"].notna().any():
        panel = panel.assign(abs_weight=lambda d: d["weight"].abs()).sort_values(
            "abs_weight", ascending=False
        )
    panel = panel.drop_duplicates(subset=["bit"]).head(max_bits)
    if panel.empty:
        return None

    images = []
    labels = []
    for _, row in panel.iterrows():
        mol = Chem.MolFromSmiles(str(row["example_smiles"]))
        if mol is None:
            continue
        bit_info: dict = {}
        AllChem.GetMorganFingerprintAsBitVect(
            mol, MORGAN_RADIUS, nBits=MORGAN_NBITS, bitInfo=bit_info
        )
        bit = int(row["bit"])
        if bit not in bit_info:
            continue
        try:
            raw = Draw.DrawMorganBit(mol, bit, bit_info, useSVG=False)
            img = Image.open(BytesIO(raw)) if isinstance(raw, bytes | bytearray) else raw
        except Exception:  # noqa: BLE001
            img = Draw.MolToImage(mol, size=(250, 200))
        images.append(img)
        w = row["weight"]
        frag = row.get("fragment_smiles") or "?"
        labels.append(f"bit {bit}: {frag}" + (f" (w={w:+.2f})" if pd.notna(w) else ""))

    if not images:
        return None

    n = len(images)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows))
    axes_arr = np.atleast_1d(axes).ravel()
    for ax in axes_arr:
        ax.axis("off")
    for ax, img, label in zip(axes_arr, images, labels, strict=False):
        ax.imshow(img)
        ax.set_title(label, fontsize=8)
        ax.axis("off")
    fig.suptitle("Figure 6. Top Morgan bits → example substructures", y=1.02, fontsize=11)
    fig.tight_layout()
    out = out_path or (FIGURES / "fig6_morgan_bit_substructures.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out)
    return out


def plot_error_scaffold_rates(
    enrichment: pd.DataFrame,
    top_n: int = 12,
    out_path: Path | None = None,
) -> Path | None:
    ensure_dirs()
    if enrichment.empty:
        return None
    top = enrichment.head(top_n).iloc[::-1].copy()
    labels = [
        s if len(s) <= 26 else s[:23] + "…" for s in top["scaffold"].astype(str)
    ]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    y = np.arange(len(top))
    ax.barh(y - 0.15, top["fp_rate"], height=0.3, label="FP rate", color="#b9770e")
    ax.barh(y + 0.15, top["fn_rate"], height=0.3, label="FN rate", color="#1b4f72")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Rate within scaffold (scaffold-split test)")
    ax.set_title("Scaffolds enriched in false positives / false negatives")
    ax.set_xlim(0, 1.05)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = out_path or (FIGURES / "qsar_error_scaffolds.png")
    fig.savefig(out, dpi=180)
    plt.close(fig)
    logger.info("Wrote %s", out)
    return out


def nearest_train_neighbors(
    X_train: np.ndarray,
    y_train: np.ndarray,
    smiles_train: list[str] | pd.Series,
    X_query: np.ndarray,
    y_query: np.ndarray,
    smiles_query: list[str] | pd.Series,
    error_tag: np.ndarray,
    classes: tuple[str, ...] = ("TP", "FP", "FN"),
    n_examples: int = 5,
    k: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """For sampled TP/FP/FN queries, find k nearest train neighbors (Tanimoto/Jaccard)."""
    from sklearn.metrics import pairwise_distances

    rng = np.random.RandomState(random_state)
    smiles_train = pd.Series(list(smiles_train)).reset_index(drop=True)
    smiles_query = pd.Series(list(smiles_query)).reset_index(drop=True)
    rows = []
    # Jaccard distance on binary fingerprints ≡ 1 − Tanimoto
    X_train_b = X_train.astype(bool)
    X_query_b = X_query.astype(bool)
    for label in classes:
        idxs = np.where(error_tag == label)[0]
        if len(idxs) == 0:
            continue
        take = min(n_examples, len(idxs))
        chosen = rng.choice(idxs, size=take, replace=False)
        dist = pairwise_distances(X_query_b[chosen], X_train_b, metric="jaccard")
        for local_i, qi in enumerate(chosen):
            order = np.argsort(dist[local_i])[:k]
            for rank, ti in enumerate(order, start=1):
                rows.append(
                    {
                        "error_class": label,
                        "query_idx": int(qi),
                        "query_smiles": smiles_query.iloc[qi],
                        "query_label": int(y_query[qi]),
                        "neighbor_rank": rank,
                        "neighbor_idx": int(ti),
                        "neighbor_smiles": smiles_train.iloc[ti],
                        "neighbor_label": int(y_train[ti]),
                        "tanimoto": float(1.0 - dist[local_i, ti]),
                    }
                )
    return pd.DataFrame(rows)


def plot_neighbor_summary(
    neighbors: pd.DataFrame,
    out_path: Path | None = None,
) -> Path | None:
    """Bar summary: mean Tanimoto of nearest train neighbor by error class."""
    ensure_dirs()
    if neighbors.empty:
        return None
    first = neighbors[neighbors["neighbor_rank"] == 1]
    if first.empty:
        return None
    means = first.groupby("error_class")["tanimoto"].mean()
    same = first.assign(same_label=lambda d: d["query_label"] == d["neighbor_label"])
    same_rate = same.groupby("error_class")["same_label"].mean()

    order = [c for c in ("TP", "FP", "FN") if c in means.index]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].bar(order, [means[c] for c in order], color=["#1e8449", "#b9770e", "#1b4f72"])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Mean Tanimoto to nearest train neighbor")
    axes[0].set_title("Structural proximity to training set")
    axes[1].bar(
        order, [same_rate.get(c, 0) for c in order], color=["#1e8449", "#b9770e", "#1b4f72"]
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Fraction with same activity label")
    axes[1].set_title("Nearest neighbor label agreement")
    fig.suptitle("Figure 5. What are TP / FP / FN near in chemical space?", y=1.02)
    fig.tight_layout()
    out = out_path or (FIGURES / "fig5_error_neighbors.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out)
    return out


def run_interpretation(
    model: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    smiles_test: list[str] | pd.Series,
    smiles_train: list[str] | pd.Series | None = None,
    top_bits: int = 40,
    n_perm_bits: int = 30,
) -> dict:
    """Fit-free interpretation on an already-trained logreg pipeline."""
    ensure_dirs()
    weights = extract_logreg_weights(model)
    weights_path = PROCESSED / "qsar_logreg_bit_weights.csv"
    weights.to_csv(weights_path, index=False)

    _, y_pred = predict_binary(model, X_test)
    enrichment = scaffold_error_enrichment(smiles_test, y_test, y_pred)
    enrich_path = PROCESSED / "qsar_error_scaffolds.csv"
    enrichment.to_csv(enrich_path, index=False)

    err = error_labels(y_test, y_pred)
    err_counts = {k: int((err == k).sum()) for k in ("TP", "FP", "FN", "TN")}

    bit_idx = weights.head(n_perm_bits)["bit"].to_numpy()
    try:
        perm = top_bit_permutation_importance(
            model, X_train, y_train, bit_indices=bit_idx
        )
        perm_path = PROCESSED / "qsar_bit_permutation_importance.csv"
        perm.to_csv(perm_path, index=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Permutation importance failed: %s", exc)
        perm = pd.DataFrame()
        perm_path = None

    fig_weights = plot_logreg_weights(weights, top_n=min(20, top_bits // 2 or 20))
    fig_errors = plot_error_scaffold_rates(enrichment)

    neighbor_path = None
    fig_neighbors = None
    if smiles_train is not None:
        try:
            neighbors = nearest_train_neighbors(
                X_train,
                y_train,
                smiles_train,
                X_test,
                y_test,
                smiles_test,
                err,
            )
            neighbor_path = PROCESSED / "qsar_error_neighbors.csv"
            neighbors.to_csv(neighbor_path, index=False)
            fig_neighbors = plot_neighbor_summary(neighbors)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Nearest-neighbor analysis failed: %s", exc)

    bit_explain_path = None
    fig_bits = None
    try:
        top_pos = weights.nlargest(4, "weight")["bit"].tolist()
        top_neg = weights.nsmallest(4, "weight")["bit"].tolist()
        explain_bits = top_pos + top_neg
        # Prefer train molecules (bit was learned there); fall back to test
        pool = list(smiles_train) if smiles_train is not None else list(smiles_test)
        bit_explain = explain_morgan_bits(pool, explain_bits, weights=weights)
        bit_explain_path = PROCESSED / "qsar_bit_substructures.csv"
        bit_explain.to_csv(bit_explain_path, index=False)
        fig_bits = plot_morgan_bit_highlights(bit_explain)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Bit→substructure mapping failed: %s", exc)

    figures = [
        str(p)
        for p in (fig_weights, fig_errors, fig_neighbors, fig_bits)
        if p is not None
    ]
    return {
        "error_counts": err_counts,
        "weights_csv": str(weights_path),
        "error_scaffolds_csv": str(enrich_path),
        "neighbors_csv": str(neighbor_path) if neighbor_path else None,
        "permutation_csv": str(perm_path) if perm_path else None,
        "bit_substructures_csv": str(bit_explain_path) if bit_explain_path else None,
        "figures": figures,
        "n_test": int(len(y_test)),
        "top_positive_bits": weights.nlargest(10, "weight")["bit"].tolist(),
        "top_negative_bits": weights.nsmallest(10, "weight")["bit"].tolist(),
    }
