"""Bemis–Murcko scaffolds for diversity stats and leakage-aware splits."""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold

from abxatlas.featurize.fingerprints import smiles_to_mol


def bemis_murcko_scaffold(smiles: str) -> str | None:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:  # noqa: BLE001
        return None


def scaffold_series(smiles: Sequence[str]) -> pd.Series:
    return pd.Series([bemis_murcko_scaffold(s) for s in smiles], dtype="object")
