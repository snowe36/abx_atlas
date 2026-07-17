"""Atlas figures (matplotlib only — CPU-friendly)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from abxatlas.featurize.fingerprints import morgan_fps
from abxatlas.featurize.scaffolds import scaffold_series
from abxatlas.paths import FIGURES, ensure_dirs


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        }
    )


def _pca_xy(
    compounds: pd.DataFrame,
    max_points: int = 5000,
) -> tuple[pd.DataFrame, np.ndarray]:
    df = compounds.dropna(subset=["smiles"]).copy()
    if len(df) > max_points:
        df = df.sample(max_points, random_state=42)
    X, mask = morgan_fps(df["smiles"].tolist())
    df = df.loc[mask].reset_index(drop=True)
    X = X[mask]
    if len(df) < 5:
        raise ValueError("Too few valid molecules for PCA")
    xy = PCA(n_components=2, random_state=42).fit_transform(X.astype(np.float64))
    return df, xy


def plot_pca_chemspace(
    compounds: pd.DataFrame,
    color_by: str = "moa_bucket",
    out_path: Path | None = None,
    max_points: int = 5000,
) -> Path:
    ensure_dirs()
    _style()
    df, xy = _pca_xy(compounds, max_points=max_points)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    if color_by not in df.columns:
        color_by = "moa_bucket"
    cats = df[color_by].fillna("unknown").astype(str)
    palette = {
        "cell_envelope": "#1b4f72",
        "other": "#b9770e",
        "unknown": "#7f8c8d",
        "True": "#1e8449",
        "False": "#922b21",
        "1": "#1e8449",
        "0": "#922b21",
    }
    for cat in sorted(cats.unique()):
        m = cats == cat
        ax.scatter(
            xy[m, 0],
            xy[m, 1],
            s=12,
            alpha=0.55,
            label=str(cat),
            c=palette.get(str(cat), None),
            edgecolors="none",
        )
    ax.set_xlabel("PCA 1 (Morgan FP)")
    ax.set_ylabel("PCA 2 (Morgan FP)")
    ax.set_title(f"Antibacterial chemical space colored by {color_by}")
    ax.legend(frameon=False, markerscale=1.5, fontsize=9)
    fig.tight_layout()
    out = out_path or (FIGURES / f"pca_by_{color_by}.png")
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_np_envelope_chemspace(
    compounds: pd.DataFrame,
    out_path: Path | None = None,
    max_points: int = 5000,
) -> Path:
    """PCA overlay: synthetic background, natural products, cell-envelope highlight."""
    ensure_dirs()
    _style()
    df, xy = _pca_xy(compounds, max_points=max_points)
    is_np = (
        df["is_natural_product"].fillna(False).astype(bool)
        if "is_natural_product" in df.columns
        else pd.Series(False, index=df.index)
    )
    is_env = (
        df["moa_bucket"].fillna("unknown").astype(str).eq("cell_envelope")
        if "moa_bucket" in df.columns
        else pd.Series(False, index=df.index)
    )

    fig, ax = plt.subplots(figsize=(8, 6.5))
    syn = ~is_np
    if syn.any():
        ax.scatter(
            xy[syn.to_numpy(), 0],
            xy[syn.to_numpy(), 1],
            s=14,
            alpha=0.35,
            marker="x",
            c="#95a5a6",
            linewidths=0.7,
            label="Synthetic / other",
        )
    if is_np.any():
        ax.scatter(
            xy[is_np.to_numpy(), 0],
            xy[is_np.to_numpy(), 1],
            s=28,
            alpha=0.7,
            marker="o",
            c="#1e8449",
            edgecolors="none",
            label="Natural product",
        )
    if is_env.any():
        ax.scatter(
            xy[is_env.to_numpy(), 0],
            xy[is_env.to_numpy(), 1],
            s=55,
            alpha=0.9,
            marker="o",
            facecolors="none",
            edgecolors="#1b4f72",
            linewidths=1.2,
            label="Cell-envelope target",
        )
    ax.set_xlabel("PCA 1 (Morgan FP)")
    ax.set_ylabel("PCA 2 (Morgan FP)")
    ax.set_title("Do natural products occupy underexplored antibacterial space?")
    ax.legend(frameon=False, markerscale=1.2, fontsize=9, loc="best")
    fig.tight_layout()
    out = out_path or (FIGURES / "pca_np_vs_envelope.png")
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_scaffold_counts(
    compounds: pd.DataFrame,
    out_path: Path | None = None,
    top_n: int = 15,
) -> Path:
    ensure_dirs()
    _style()
    scaf = scaffold_series(compounds["smiles"].tolist())
    counts = scaf.value_counts().head(top_n)
    labels = [s if len(s) <= 28 else s[:25] + "…" for s in counts.index.astype(str)]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(range(len(counts))[::-1], counts.values, color="#1b4f72")
    ax.set_yticks(range(len(counts))[::-1])
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Number of compounds")
    ax.set_title("Top Bemis–Murcko scaffolds in antibacterial set")
    fig.tight_layout()
    out = out_path or (FIGURES / "top_scaffolds.png")
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_np_vs_synthetic_moa(
    compounds: pd.DataFrame,
    out_path: Path | None = None,
) -> Path:
    ensure_dirs()
    _style()
    df = compounds.copy()
    df["np_label"] = df["is_natural_product"].fillna(False).map(
        {True: "Natural product", False: "Synthetic / other"}
    )
    ct = pd.crosstab(df["moa_bucket"], df["np_label"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ct.plot(kind="bar", ax=ax, color=["#1e8449", "#5d6d7e"], width=0.75)
    ax.set_xlabel("MoA bucket")
    ax.set_ylabel("Compounds")
    ax.set_title("Natural product vs synthetic by target bucket")
    ax.legend(frameon=False)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    out = out_path or (FIGURES / "np_vs_synthetic_by_moa.png")
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_gram_label_balance(
    compounds: pd.DataFrame,
    out_path: Path | None = None,
) -> Path:
    ensure_dirs()
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    for ax, col, title in [
        (axes[0], "gram_neg_active", "Gram-negative activity"),
        (axes[1], "gram_pos_active", "Gram-positive activity"),
    ]:
        if col not in compounds.columns:
            ax.set_axis_off()
            continue
        s = compounds[col].dropna().astype(int).value_counts().sort_index()
        ax.bar(
            ["Inactive", "Active"],
            [s.get(0, 0), s.get(1, 0)],
            color=["#922b21", "#1e8449"],
        )
        ax.set_title(title)
        ax.set_ylabel("Compounds")
    fig.suptitle("Binary label balance (pChEMBL ≥ 5)", y=1.02)
    fig.tight_layout()
    out = out_path or (FIGURES / "gram_label_balance.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out
