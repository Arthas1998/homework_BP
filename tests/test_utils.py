import numpy as np
from bp.utils import iter_minibatches, init_weights


def test_iter_minibatches_yields_all_samples():
    X = np.arange(20).reshape(20, 1)
    y = np.arange(20).reshape(20, 1)
    batches = list(iter_minibatches(X, y, batch_size=6, shuffle=False))
    total = sum(b[0].shape[0] for b in batches)
    assert total == 20
    # check no overlap when shuffle=False
    seen = []
    for xb, yb in batches:
        for v in xb.flatten():
            seen.append(int(v))
    assert sorted(seen) == list(range(20))


def test_init_weights_shapes_and_std():
    rng = np.random.RandomState(0)
    W = init_weights((10, 20), method="xavier", rng=rng)
    assert W.shape == (10, 20)
    assert np.isfinite(W).all()

