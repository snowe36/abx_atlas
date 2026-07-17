import numpy as np

from abxatlas.models.interpret import error_labels, nearest_train_neighbors


def test_nearest_neighbors_shape():
    X_train = np.array([[1, 1, 0, 0], [1, 1, 0, 1], [0, 0, 1, 1]], dtype=float)
    y_train = np.array([1, 1, 0])
    smiles_train = ["CCO", "CCO", "c1ccccc1"]
    X_test = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=float)
    y_test = np.array([1, 0])
    smiles_test = ["CCN", "c1ccccc1O"]
    err = error_labels(y_test, np.array([1, 1]))  # TP, FP
    nn = nearest_train_neighbors(
        X_train,
        y_train,
        smiles_train,
        X_test,
        y_test,
        smiles_test,
        err,
        n_examples=2,
        k=2,
    )
    assert len(nn) >= 2
    assert {"error_class", "tanimoto", "neighbor_smiles"} <= set(nn.columns)
