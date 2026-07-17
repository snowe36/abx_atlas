from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
RESOURCES = Path(__file__).resolve().parent / "resources"


def ensure_dirs() -> None:
    for path in (RAW, PROCESSED, FIGURES):
        path.mkdir(parents=True, exist_ok=True)
