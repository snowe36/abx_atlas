import numpy as np

from abxatlas.featurize.fingerprints import morgan_fps, smiles_to_mol
from abxatlas.featurize.scaffolds import bemis_murcko_scaffold


def test_morgan_shape():
    X, mask = morgan_fps(["CCO", "c1ccccc1", "not_a_smiles"])
    assert X.shape == (3, 2048)
    assert mask.tolist() == [True, True, False]


def test_scaffold_benzene():
    scaf = bemis_murcko_scaffold("Cc1ccccc1")
    assert scaf is not None
    assert smiles_to_mol(scaf) is not None


def test_fp_deterministic():
    X1, _ = morgan_fps(["CCO"])
    X2, _ = morgan_fps(["CCO"])
    assert np.array_equal(X1, X2)
