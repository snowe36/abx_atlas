"""Target / organism / NP annotation helpers."""

from __future__ import annotations

import re

import pandas as pd

from abxatlas.config import load_keywords


def _normalize(text: str | None) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    return str(text).strip().lower()


def bucket_target(target_name: str | None, keywords: dict | None = None) -> str:
    """Return cell_envelope | other | unknown from preferred target name."""
    kw = keywords or load_keywords()
    name = _normalize(target_name)
    if not name:
        return "unknown"

    for excl in kw.get("exclude_from_envelope", []):
        if excl.lower() in name:
            return "other"

    for hit in kw.get("cell_envelope", []):
        if hit.lower() in name:
            return "cell_envelope"

    # Named molecular targets that are not envelope → other; organism-level → unknown
    if name and name not in {"unchecked", "no target assigned"}:
        # Heuristic: enzyme/protein-like names count as other MoA
        if any(
            token in name
            for token in (
                "ase",
                "kinase",
                "reductase",
                "synthase",
                "polymerase",
                "gyrase",
                "topoisomerase",
                "ribosom",
                "protein",
                "receptor",
                "channel",
                "transporter",
            )
        ):
            return "other"
    return "unknown"


def gram_stain(organism: str | None, keywords: dict | None = None) -> str:
    """Return gram_negative | gram_positive | unknown."""
    kw = keywords or load_keywords()
    org = _normalize(organism)
    if not org:
        return "unknown"
    for name in kw.get("gram_negative_organisms", []):
        if name.lower() in org:
            return "gram_negative"
    for name in kw.get("gram_positive_organisms", []):
        if name.lower() in org:
            return "gram_positive"
    return "unknown"


def annotate_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add moa_bucket, gram_stain, and coerce natural_product flag."""
    out = df.copy()
    kw = load_keywords()
    target_col = "target_name" if "target_name" in out.columns else "pref_name"
    if target_col not in out.columns:
        out["target_name"] = None
        target_col = "target_name"
    out["moa_bucket"] = out[target_col].map(lambda x: bucket_target(x, kw))
    org_col = "organism" if "organism" in out.columns else "assay_organism"
    if org_col not in out.columns:
        out["organism"] = None
        org_col = "organism"
    out["gram_stain"] = out[org_col].map(lambda x: gram_stain(x, kw))

    if "natural_product" in out.columns:
        out["is_natural_product"] = out["natural_product"].map(_as_bool)
    else:
        out["is_natural_product"] = False
    return out


def _as_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


_NP_HINT = re.compile(
    r"\b(natural product|from natural|isolated from|secondary metabolite)\b",
    re.I,
)


def np_hint_from_text(text: str | None) -> bool:
    if not text:
        return False
    return bool(_NP_HINT.search(str(text)))
