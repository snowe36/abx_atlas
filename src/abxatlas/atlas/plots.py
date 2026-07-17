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


def plot_fig1_chemspace_atlas(
    compounds: pd.DataFrame,
    out_path: Path | None = None,
    max_points: int = 5000,
) -> Path:
    """Figure 1: three-panel PCA (Gram label, Gram− activity, MoA bucket)."""
    ensure_dirs()
    _style()
    df, xy = _pca_xy(compounds, max_points=max_points)

    # Derive a Gram stain view from available labels
    gram = pd.Series("unknown", index=df.index, dtype=object)
    if "has_gram_neg_assay" in df.columns and "has_gram_pos_assay" in df.columns:
        gn = df["has_gram_neg_assay"].fillna(False).astype(bool)
        gp = df["has_gram_pos_assay"].fillna(False).astype(bool)
        gram.loc[gn & ~gp] = "Gram− only"
        gram.loc[gp & ~gn] = "Gram+ only"
        gram.loc[gn & gp] = "Both"
        gram.loc[~gn & ~gp] = "Neither / unknown"

    panels = [
        (
            "Assay organism class",
            gram,
            {
                "Gram− only": "#1b4f72",
                "Gram+ only": "#b9770e",
                "Both": "#1e8449",
                "Neither / unknown": "#bdc3c7",
            },
        ),
        (
            "Gram− activity (pChEMBL ≥ 5)",
            df["gram_neg_active"].map({1: "Active", 0: "Inactive", 1.0: "Active", 0.0: "Inactive"}).fillna("No label")
            if "gram_neg_active" in df.columns
            else pd.Series("No label", index=df.index),
            {"Active": "#1e8449", "Inactive": "#922b21", "No label": "#bdc3c7"},
        ),
        (
            "MoA bucket",
            df["moa_bucket"].fillna("unknown").astype(str)
            if "moa_bucket" in df.columns
            else pd.Series("unknown", index=df.index),
            {
                "cell_envelope": "#1b4f72",
                "other": "#b9770e",
                "unknown": "#7f8c8d",
            },
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharex=True, sharey=True)
    for ax, (title, cats, palette) in zip(axes, panels):
        cats = cats.astype(str)
        for cat in sorted(cats.unique()):
            m = cats == cat
            ax.scatter(
                xy[m.to_numpy(), 0],
                xy[m.to_numpy(), 1],
                s=10,
                alpha=0.5,
                c=palette.get(cat, "#7f8c8d"),
                edgecolors="none",
                label=cat,
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("PCA 1")
        ax.legend(frameon=False, markerscale=1.8, fontsize=8, loc="best")
    axes[0].set_ylabel("PCA 2 (Morgan FP)")
    fig.suptitle(
        "Figure 1. Antibacterial chemical space (Morgan FP PCA)",
        y=1.02,
        fontsize=13,
    )
    fig.tight_layout()
    out = out_path or (FIGURES / "fig1_chemspace_atlas.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_fig2_scaffold_diversity(
    compounds: pd.DataFrame,
    out_path: Path | None = None,
) -> Path:
    """Figure 2: compound vs scaffold counts + frequency distribution."""
    ensure_dirs()
    _style()
    scaf = scaffold_series(compounds["smiles"].tolist())
    n_comp = int(len(compounds))
    n_scaf = int(scaf.nunique(dropna=True))
    freq = scaf.value_counts(dropna=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(
        ["Compounds", "Unique scaffolds"],
        [n_comp, n_scaf],
        color=["#1b4f72", "#5d6d7e"],
        width=0.55,
    )
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Scaffold ratio = {n_scaf / n_comp:.2f}" if n_comp else "")
    for i, v in enumerate([n_comp, n_scaf]):
        axes[0].text(i, v, f" {v:,}", ha="center", va="bottom", fontsize=10)

    # Log-scale histogram of scaffold frequencies
    if len(freq):
        axes[1].hist(
            freq.values,
            bins=min(40, max(10, int(np.sqrt(len(freq))))),
            color="#1b4f72",
            edgecolor="white",
            log=True,
        )
    axes[1].set_xlabel("Compounds per scaffold")
    axes[1].set_ylabel("Number of scaffolds (log)")
    axes[1].set_title("Most scaffolds are rare — hard for random splits")
    fig.suptitle("Figure 2. Bemis–Murcko scaffold diversity", y=1.02, fontsize=13)
    fig.tight_layout()
    out = out_path or (FIGURES / "fig2_scaffold_diversity.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out
