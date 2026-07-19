"""RDKit → PyG graphs (optional torch deps)."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from abxatlas.featurize.fingerprints import smiles_to_mol

logger = logging.getLogger(__name__)

# Common elements in bioactive small molecules; anything else falls into the
# trailing "other" one-hot slot so unseen elements degrade gracefully.
ATOM_VOCAB = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se"]
HYBRIDIZATION_VOCAB = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]
BOND_VOCAB = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]

# +1 per vocab for an "other/unknown" slot; trailing ints are numeric scalars.
ATOM_FEATURE_DIM = (len(ATOM_VOCAB) + 1) + (len(HYBRIDIZATION_VOCAB) + 1) + 5
BOND_FEATURE_DIM = (len(BOND_VOCAB) + 1) + 2


def _one_hot(value: str, vocab: Sequence[str]) -> list[float]:
    vec = [0.0] * (len(vocab) + 1)
    try:
        vec[vocab.index(value)] = 1.0
    except ValueError:
        vec[-1] = 1.0
    return vec


def _atom_features(atom) -> list[float]:
    feats = _one_hot(atom.GetSymbol(), ATOM_VOCAB)
    hyb = str(atom.GetHybridization()).replace("HybridizationType.", "")
    feats += _one_hot(hyb, HYBRIDIZATION_VOCAB)
    feats.append(atom.GetDegree() / 4.0)
    feats.append(float(atom.GetFormalCharge()))
    feats.append(1.0 if atom.GetIsAromatic() else 0.0)
    feats.append(1.0 if atom.IsInRing() else 0.0)
    feats.append(atom.GetTotalNumHs() / 4.0)
    return feats


def _bond_features(bond) -> list[float]:
    bond_type = str(bond.GetBondType()).replace("BondType.", "")
    feats = _one_hot(bond_type, BOND_VOCAB)
    feats.append(1.0 if bond.GetIsConjugated() else 0.0)
    feats.append(1.0 if bond.IsInRing() else 0.0)
    return feats


def mol_to_graph(smiles: str):
    """RDKit SMILES -> torch_geometric.data.Data, or None if invalid/unparseable.

    Requires torch/torch_geometric; imported lazily so plain CPU installs of
    this repo (without the `gpu` extra) never need them.
    """
    import torch
    from torch_geometric.data import Data

    mol = smiles_to_mol(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    try:
        x = torch.tensor([_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)
        edges: list[tuple[int, int]] = []
        edge_feats: list[list[float]] = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bf = _bond_features(bond)
            edges.append((i, j))
            edge_feats.append(bf)
            edges.append((j, i))
            edge_feats.append(bf)
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, BOND_FEATURE_DIM), dtype=torch.float32)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Graph build failed for %s: %s", smiles, exc)
        return None


def smiles_to_graphs(smiles: Sequence[str]) -> tuple[list, np.ndarray]:
    """Return (graphs, valid_mask), mirroring the morgan_fps(...) contract.

    Unlike morgan_fps, invalid entries are simply omitted from `graphs`
    (rather than kept as zero rows) — `mask` tells the caller which original
    positions survived, in the same relative order.
    """
    n = len(smiles)
    mask = np.zeros(n, dtype=bool)
    graphs = []
    for i, smi in enumerate(smiles):
        g = mol_to_graph(smi)
        if g is not None:
            graphs.append(g)
            mask[i] = True
    bad = int((~mask).sum())
    if bad:
        logger.warning("Dropped/invalid SMILES for graphs: %d / %d", bad, n)
    return graphs, mask
