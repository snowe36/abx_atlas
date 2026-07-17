"""Thin ChEMBL antibacterial activity download (independent of curated forks)."""

from __future__ import annotations

import logging
import time
from typing import Iterable

import pandas as pd
from chembl_webresource_client.new_client import new_client
from chembl_webresource_client.settings import Settings
from tqdm import tqdm

from abxatlas.config import load_keywords
from abxatlas.paths import RAW, ensure_dirs

logger = logging.getLogger(__name__)

# Standard antimicrobial potency types we keep
ACTIVITY_TYPES = ("MIC", "IC50", "Ki", "EC50", "IZ", "Potency")

# Priority pathogens for a laptop-scale first pass (still covers Gram+/−)
PRIORITY_ORGANISMS = [
    "Escherichia coli",
    "Pseudomonas aeruginosa",
    "Klebsiella pneumoniae",
    "Acinetobacter baumannii",
    "Staphylococcus aureus",
    "Streptococcus pneumoniae",
    "Enterococcus faecium",
    "Mycobacterium tuberculosis",
]


def _configure_client(page_size: int = 1000) -> None:
    """Raise ChEMBL client page size (default 20 is painfully slow)."""
    settings = Settings.Instance()
    settings.MAX_LIMIT = page_size
    settings.CACHING = True


def _organism_queries(keywords: dict | None = None, priority_only: bool = False) -> list[str]:
    if priority_only:
        return list(PRIORITY_ORGANISMS)
    kw = keywords or load_keywords()
    return list(kw.get("gram_negative_organisms", [])) + list(
        kw.get("gram_positive_organisms", [])
    )


def fetch_organism_activities(
    organisms: Iterable[str] | None = None,
    max_per_organism: int | None = 8000,
    activity_types: Iterable[str] = ACTIVITY_TYPES,
    sleep_s: float = 0.05,
    priority_only: bool = False,
) -> pd.DataFrame:
    """Pull organism-level antibacterial activities from ChEMBL.

    Caps per organism keep a laptop download finishable; raise or set None for fuller pulls.
    """
    ensure_dirs()
    _configure_client(1000)
    orgs = list(organisms) if organisms is not None else _organism_queries(priority_only=priority_only)
    activity = new_client.activity
    frames: list[pd.DataFrame] = []

    for org in tqdm(orgs, desc="ChEMBL organisms"):
        try:
            query = activity.filter(
                target_organism__iexact=org,
                pchembl_value__isnull=False,
                standard_type__in=list(activity_types),
            ).only(
                [
                    "molecule_chembl_id",
                    "canonical_smiles",
                    "standard_type",
                    "standard_value",
                    "standard_units",
                    "pchembl_value",
                    "assay_chembl_id",
                    "assay_description",
                    "target_chembl_id",
                    "target_organism",
                    "target_pref_name",
                    "bao_label",
                    "document_year",
                    "data_validity_comment",
                ]
            )
            rows = []
            for i, rec in enumerate(query):
                rows.append(rec)
                if max_per_organism is not None and i + 1 >= max_per_organism:
                    break
            if rows:
                frames.append(pd.DataFrame(rows))
                logger.info("%s: %d activities", org, len(rows))
            else:
                logger.info("%s: 0 activities", org)
        except Exception as exc:  # noqa: BLE001 — keep download resilient
            logger.warning("Failed organism %s: %s", org, exc)
        time.sleep(sleep_s)

    if not frames:
        raise RuntimeError(
            "No ChEMBL activities downloaded. Check network / ChEMBL API availability."
        )
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(
        columns={
            "target_organism": "organism",
            "target_pref_name": "target_name",
            "canonical_smiles": "smiles",
        }
    )
    return df


def fetch_molecule_np_flags(
    chembl_ids: Iterable[str],
    max_ids: int = 5000,
    chunk_size: int = 50,
) -> pd.DataFrame:
    """Fetch natural_product flags for a capped set of molecule ChEMBL IDs."""
    _configure_client(1000)
    molecule = new_client.molecule
    ids = sorted({c for c in chembl_ids if c})[:max_ids]
    records = []
    for start in tqdm(range(0, len(ids), chunk_size), desc="NP flags"):
        chunk = ids[start : start + chunk_size]
        try:
            mols = list(
                molecule.filter(molecule_chembl_id__in=chunk).only(
                    [
                        "molecule_chembl_id",
                        "natural_product",
                        "molecule_type",
                        "first_approval",
                    ]
                )
            )
            records.extend(mols)
        except Exception:
            for mid in chunk:
                try:
                    mols = list(
                        molecule.filter(molecule_chembl_id=mid).only(
                            ["molecule_chembl_id", "natural_product"]
                        )
                    )
                    if mols:
                        records.append(mols[0])
                except Exception as exc:  # noqa: BLE001
                    logger.debug("NP flag failed for %s: %s", mid, exc)
        time.sleep(0.05)
    if not records:
        return pd.DataFrame(columns=["molecule_chembl_id", "natural_product"])
    return pd.DataFrame(records)


def download_antibacterial(
    out_path=None,
    max_per_organism: int | None = 8000,
    fetch_np: bool = True,
    priority_only: bool = True,
) -> pd.DataFrame:
    """Download raw antibacterial activities (+ optional NP flags) to parquet.

    By default pulls a priority ESKAPE-ish organism set for a finishable laptop run.
    Pass priority_only=False for the full organism list in target_keywords.yaml.
    """
    ensure_dirs()
    out = out_path or (RAW / "chembl_antibacterial_raw.parquet")
    df = fetch_organism_activities(
        max_per_organism=max_per_organism,
        priority_only=priority_only,
    )
    if fetch_np and "molecule_chembl_id" in df.columns:
        unique_ids = df["molecule_chembl_id"].dropna().unique().tolist()
        np_df = fetch_molecule_np_flags(unique_ids)
        if not np_df.empty:
            df = df.merge(np_df, on="molecule_chembl_id", how="left")
    df.to_parquet(out, index=False)
    logger.info("Wrote %s (%d rows)", out, len(df))
    return df
