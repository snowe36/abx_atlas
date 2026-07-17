from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from abxatlas.paths import RESOURCES

# pChEMBL >= 5 ≈ IC50/MIC <= 10 µM (standard chemogenomics cutoff)
ACTIVE_PCHEMBL = 5.0
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048
RANDOM_STATE = 42


@lru_cache(maxsize=1)
def load_keywords(path: Path | None = None) -> dict:
    yaml_path = path or (RESOURCES / "target_keywords.yaml")
    with open(yaml_path, encoding="utf-8") as f:
        return yaml.safe_load(f)
