"""Filter, binarize, and compound-level aggregation of antibacterial activities."""

from __future__ import annotations

import logging

import pandas as pd

from abxatlas.config import ACTIVE_PCHEMBL
from abxatlas.data.annotate import annotate_frame
from abxatlas.paths import PROCESSED, RAW, ensure_dirs

logger = logging.getLogger(__name__)

CURATED_PATH = PROCESSED / "antibacterial_curated.parquet"
COMPOUND_PATH = PROCESSED / "compounds.parquet"


def _best_pchembl(series: pd.Series) -> float:
    return float(series.max())


def curate_activities(
    raw: pd.DataFrame | None = None,
    active_pchembl: float = ACTIVE_PCHEMBL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (assay-level curated, compound-level table)."""
    ensure_dirs()
    if raw is None:
        raw_path = RAW / "chembl_antibacterial_raw.parquet"
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Missing {raw_path}. Run: abx-download  (or scripts/download_chembl.py)"
            )
        raw = pd.read_parquet(raw_path)

    df = raw.copy()
    df = df.dropna(subset=["smiles", "pchembl_value", "molecule_chembl_id"])
    df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df = df.dropna(subset=["pchembl_value"])
    if "data_validity_comment" in df.columns:
        # Drop explicitly flagged invalid points when present
        bad = df["data_validity_comment"].fillna("").astype(str).str.len() > 0
        # Keep Outside typical range cautiously; drop only clear failures
        drop_mask = df["data_validity_comment"].fillna("").str.contains(
            "Potential transcription error|Non standard unit", case=False, na=False
        )
        df = df.loc[~drop_mask].copy()

    df = annotate_frame(df)
    df["active"] = df["pchembl_value"] >= active_pchembl
    df["active_pchembl_cutoff"] = active_pchembl

    # Compound × organism aggregation: max pChEMBL (most potent reported)
    group_cols = ["molecule_chembl_id", "smiles", "organism", "gram_stain"]
    agg_kwargs: dict = {
        "pchembl_max": ("pchembl_value", _best_pchembl),
        "moa_bucket": ("moa_bucket", _majority_or_envelope),
        "is_natural_product": ("is_natural_product", "max"),
    }
    if "assay_chembl_id" in df.columns:
        agg_kwargs["n_assays"] = ("assay_chembl_id", "nunique")
    else:
        agg_kwargs["n_assays"] = ("pchembl_value", "count")
    if "document_year" in df.columns:
        agg_kwargs["document_year"] = ("document_year", "min")

    agg = df.groupby(group_cols, dropna=False).agg(**agg_kwargs).reset_index()
    agg["active"] = agg["pchembl_max"] >= active_pchembl

    # One row per compound for chemspace / QSAR primary task
    compounds = _compound_table(agg, df)

    df.to_parquet(CURATED_PATH, index=False)
    compounds.to_parquet(COMPOUND_PATH, index=False)
    n_gn_active = 0
    if "gram_neg_active" in compounds.columns:
        n_gn_active = int((compounds["gram_neg_active"] == 1).sum())
    logger.info(
        "Curated %d assay rows → %d compound rows (%d active Gram−)",
        len(df),
        len(compounds),
        n_gn_active,
    )
    return df, compounds


def _majority_or_envelope(series: pd.Series) -> str:
    vals = series.dropna().astype(str)
    if (vals == "cell_envelope").any():
        return "cell_envelope"
    if (vals == "other").any():
        return "other"
    return "unknown"


def _compound_table(agg: pd.DataFrame, assay_df: pd.DataFrame) -> pd.DataFrame:
    """Build molecule-level labels for Gram+/− and MoA buckets.

    Pre-groups assay_df by molecule once (O(n)) instead of re-scanning the
    whole assay-level frame per molecule (O(n_molecules * n_assay_rows)) —
    matters once the curated set is tens of thousands of compounds.
    """
    assay_moa_by_mol = assay_df.groupby("molecule_chembl_id")["moa_bucket"]
    rows = []
    for mid, g in agg.groupby("molecule_chembl_id"):
        smiles = g["smiles"].iloc[0]
        gn = g[g["gram_stain"] == "gram_negative"]
        gp = g[g["gram_stain"] == "gram_positive"]
        gram_neg_active = int(bool(gn["active"].any())) if len(gn) else None
        gram_pos_active = int(bool(gp["active"].any())) if len(gp) else None
        # Only label Gram− if we have at least one Gram− assay
        has_gn = len(gn) > 0
        has_gp = len(gp) > 0

        # Prefer assay-level envelope evidence; fall back to the agg-level bucket.
        if mid in assay_moa_by_mol.groups:
            moa = _majority_or_envelope(assay_moa_by_mol.get_group(mid))
        else:
            moa = _majority_or_envelope(g["moa_bucket"])

        is_np = bool(g["is_natural_product"].max())
        year = None
        if "document_year" in g.columns:
            years = pd.to_numeric(g["document_year"], errors="coerce").dropna()
            year = int(years.min()) if len(years) else None

        rows.append(
            {
                "molecule_chembl_id": mid,
                "smiles": smiles,
                "is_natural_product": is_np,
                "moa_bucket": moa,
                "has_gram_neg_assay": has_gn,
                "has_gram_pos_assay": has_gp,
                "gram_neg_active": gram_neg_active if has_gn else pd.NA,
                "gram_pos_active": gram_pos_active if has_gp else pd.NA,
                "pchembl_max_any": float(g["pchembl_max"].max()),
                "n_organisms": g["organism"].nunique(),
                "document_year_min": year,
            }
        )
    return pd.DataFrame(rows)


def load_curated() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not CURATED_PATH.exists() or not COMPOUND_PATH.exists():
        return curate_activities()
    return pd.read_parquet(CURATED_PATH), pd.read_parquet(COMPOUND_PATH)
