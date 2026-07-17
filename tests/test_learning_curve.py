import numpy as np

from abxatlas.models.learning_curve import scaffold_learning_curve


def test_learning_curve_grows_with_fraction():
    rng = np.random.RandomState(0)
    n = 200
    X = rng.randint(0, 2, size=(n, 64)).astype(float)
    y = rng.randint(0, 2, size=n)
    # Make y somewhat predictable from first bits
    y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
    train_idx = np.arange(160)
    test_idx = np.arange(160, 200)
    curve = scaffold_learning_curve(
        X, y, train_idx, test_idx, fractions=(0.25, 0.5, 1.0), random_state=0
    )
    assert not curve.empty
    assert set(curve["model_name"]) <= {"logreg", "rf"}
    assert curve["n_train"].min() < curve["n_train"].max()
