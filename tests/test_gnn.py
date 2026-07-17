import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from abxatlas.featurize.graphs import ATOM_FEATURE_DIM, smiles_to_graphs  # noqa: E402
from abxatlas.models.gnn import evaluate_gnn_split  # noqa: E402

SMILES = [
    "C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC", "CCCCCCC", "CCCCCCCC",
    "CO", "CCO", "CCCO", "CCCCO", "CCCCCO", "CCCCCCO", "CCCCCCCO", "CCCCCCCCO",
    "c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "Clc1ccccc1", "Fc1ccccc1", "Brc1ccccc1",
    "c1ccncc1", "Cc1ccncc1", "c1ccoc1", "c1ccsc1",
    "C1CCCCC1", "CC1CCCCC1", "C1CCCC1", "CC1CCCC1",
    "CC(C)C", "CC(C)CC", "CC(=O)C", "CC(=O)CC", "CC(=O)O", "CCC(=O)O",
    "NCC", "NCCC", "OCC(O)CO", "ClCCl",
]


def test_smiles_to_graphs_shape():
    graphs, mask = smiles_to_graphs(SMILES + ["not_a_smiles"])
    assert mask.tolist() == [True] * len(SMILES) + [False]
    assert len(graphs) == len(SMILES)
    assert graphs[0].x.shape[1] == ATOM_FEATURE_DIM
    assert graphs[0].edge_index.shape[0] == 2


def test_evaluate_gnn_split_smoke():
    graphs, mask = smiles_to_graphs(SMILES)
    assert mask.all()
    y = np.array([i % 2 for i in range(len(SMILES))])
    train_idx = np.arange(30)
    test_idx = np.arange(30, len(SMILES))

    results = evaluate_gnn_split(
        graphs,
        y,
        train_idx,
        test_idx,
        split_name="smoke",
        epochs=2,
        config={"hidden_dim": 16, "num_layers": 2},
    )
    assert len(results) == 1
    r = results[0]
    assert r.model_name == "gnn"
    assert r.split_name == "smoke"
    assert r.n_train == 30
    assert r.n_test == len(SMILES) - 30
    assert 0.0 <= r.roc_auc <= 1.0
    assert not np.isnan(r.roc_auc)
