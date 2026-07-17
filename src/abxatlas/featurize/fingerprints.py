"""RDKit Morgan fingerprints and basic physchem descriptors."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors

from abxatlas.config import MORGAN_NBITS, MORGAN_RADIUS

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)


def smiles_to_mol(smiles: str):
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    return mol


def morgan_fp_vector(mol, radius: int = MORGAN_RADIUS, n_bits: int = MORGAN_NBITS) -> np.ndarray | None:
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def morgan_fps(
    smiles: Sequence[str],
    radius: int = MORGAN_RADIUS,
    n_bits: int = MORGAN_NBITS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X [n, n_bits], valid_mask [n]). Invalid SMILES are zero rows with mask False."""
    n = len(smiles)
    X = np.zeros((n, n_bits), dtype=np.uint8)
    mask = np.zeros(n, dtype=bool)
    for i, smi in enumerate(smiles):
        mol = smiles_to_mol(smi)
        vec = morgan_fp_vector(mol, radius=radius, n_bits=n_bits)
        if vec is not None:
            X[i] = vec
            mask[i] = True
    bad = int((~mask).sum())
    if bad:
        logger.warning("Dropped/invalid SMILES: %d / %d", bad, n)
    return X, mask


def descriptor_frame(smiles: Sequence[str]) -> pd.DataFrame:
    rows = []
    for smi in smiles:
        mol = smiles_to_mol(smi)
        if mol is None:
            rows.append(
                {
                    "mw": np.nan,
                    "logp": np.nan,
                    "tpsa": np.nan,
                    "hbd": np.nan,
                    "hba": np.nan,
                    "rotb": np.nan,
                }
            )
            continue
        rows.append(
            {
                "mw": Descriptors.MolWt(mol),
                "logp": Descriptors.MolLogP(mol),
                "tpsa": Descriptors.TPSA(mol),
                "hbd": Descriptors.NumHDonors(mol),
                "hba": Descriptors.NumHAcceptors(mol),
                "rotb": Descriptors.NumRotatableBonds(mol),
            }
        )
    return pd.DataFrame(rows)
