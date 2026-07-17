import numpy as np

from abxatlas.models.splits import random_split_indices, scaffold_split_indices


def test_random_split_sizes():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    tr, te = random_split_indices(len(y), test_size=0.3, y=y)
    assert len(tr) + len(te) == len(y)
    assert len(set(tr) & set(te)) == 0


def test_scaffold_split_no_overlap():
    # Two chemotypes: benzene-like vs pyridine-like cores with substitutions
    smiles = [
        "c1ccccc1",
        "Cc1ccccc1",
        "Clc1ccccc1",
        "c1ccncc1",
        "Cc1ccncc1",
        "Clc1ccncc1",
    ]
    tr, te = scaffold_split_indices(smiles, test_size=0.4, random_state=0)
    assert len(tr) + len(te) == len(smiles)
    assert len(set(tr) & set(te)) == 0
