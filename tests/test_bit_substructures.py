"""Tests for Morgan bit → substructure mapping."""

import numpy as np
import pandas as pd

from abxatlas.models.interpret import explain_morgan_bits, plot_morgan_bit_highlights


def test_explain_morgan_bits_finds_fragments():
    smiles = [
        "c1ccccc1",
        "Cc1ccccc1",
        "c1ccncc1",
        "CCn1cc(C(=O)O)c(=O)c2cc(F)c(N3CCNCC3)cc21",
    ]
    # Bit 0 may or may not be set; discover a bit that is set on benzene
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from abxatlas.config import MORGAN_NBITS, MORGAN_RADIUS

    mol = Chem.MolFromSmiles("c1ccccc1")
    bit_info = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_NBITS, bitInfo=bit_info)
    bits = list(bit_info.keys())[:3]
    assert bits

    weights = pd.DataFrame({"bit": bits, "weight": [1.0, -0.5, 0.2][: len(bits)]})
    explained = explain_morgan_bits(smiles, bits, weights=weights)
    assert len(explained) == len(bits)
    assert explained["fragment_smiles"].notna().any()
    assert explained["example_smiles"].notna().any()


def test_plot_morgan_bit_highlights(tmp_path):
    smiles = ["c1ccccc1", "Cc1ccccc1", "Clc1ccccc1"]
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from abxatlas.config import MORGAN_NBITS, MORGAN_RADIUS

    mol = Chem.MolFromSmiles(smiles[0])
    bit_info = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_NBITS, bitInfo=bit_info)
    bits = list(bit_info.keys())[:4]
    weights = pd.DataFrame(
        {"bit": bits, "weight": np.linspace(1.0, -1.0, len(bits))}
    )
    explained = explain_morgan_bits(smiles, bits, weights=weights)
    out = tmp_path / "bits.png"
    path = plot_morgan_bit_highlights(explained, out_path=out, max_bits=4)
    assert path is not None
    assert out.exists()
    assert out.stat().st_size > 0
