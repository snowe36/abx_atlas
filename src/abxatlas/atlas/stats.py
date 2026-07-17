"""Chemical-space summary statistics."""

from __future__ import annotations

import pandas as pd

from abxatlas.featurize.scaffolds import scaffold_series


def scaffold_diversity(df: pd.DataFrame, smiles_col: str = "smiles") -> dict:
    scaffolds = scaffold_series(df[smiles_col].tolist())
    n = len(df)
    n_scaf = scaffolds.nunique(dropna=True)
    top = scaffolds.value_counts(dropna=True).head(10)
    return {
        "n_compounds": n,
        "n_scaffolds": int(n_scaf),
        "scaffold_ratio": float(n_scaf / n) if n else 0.0,
        "top_scaffolds": top.to_dict(),
        "scaffolds": scaffolds,
    }


def group_counts(df: pd.DataFrame) -> dict[str, pd.Series]:
    out = {}
    if "moa_bucket" in df.columns:
        out["moa_bucket"] = df["moa_bucket"].value_counts(dropna=False)
    if "is_natural_product" in df.columns:
        out["is_natural_product"] = df["is_natural_product"].value_counts(dropna=False)
    if "gram_neg_active" in df.columns:
        out["gram_neg_active"] = df["gram_neg_active"].value_counts(dropna=False)
    if "gram_pos_active" in df.columns:
        out["gram_pos_active"] = df["gram_pos_active"].value_counts(dropna=False)
    return out


def summarize_atlas(compounds: pd.DataFrame) -> pd.DataFrame:
    """Return a small printable summary table."""
    rows = []
    n = len(compounds)
    rows.append({"metric": "n_compounds", "value": n})
    div = scaffold_diversity(compounds)
    rows.append({"metric": "n_scaffolds", "value": div["n_scaffolds"]})
    rows.append({"metric": "scaffold_ratio", "value": round(div["scaffold_ratio"], 3)})
    if "is_natural_product" in compounds.columns:
        np_n = int(compounds["is_natural_product"].fillna(False).sum())
        rows.append({"metric": "n_natural_products", "value": np_n})
        rows.append(
            {"metric": "frac_natural_products", "value": round(np_n / n, 3) if n else 0}
        )
    if "moa_bucket" in compounds.columns:
        for bucket, cnt in compounds["moa_bucket"].value_counts().items():
            rows.append({"metric": f"moa_{bucket}", "value": int(cnt)})
    gn = compounds.dropna(subset=["gram_neg_active"]) if "gram_neg_active" in compounds else None
    if gn is not None and len(gn):
        rows.append({"metric": "n_with_gram_neg_label", "value": len(gn)})
        rows.append(
            {
                "metric": "gram_neg_active_rate",
                "value": round(float(gn["gram_neg_active"].astype(float).mean()), 3),
            }
        )
    return pd.DataFrame(rows)
