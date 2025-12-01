import numpy as np
from bp.network import ThreeLayerNet


def test_forward_shape():
    net = ThreeLayerNet(input_dim=3, hidden_dim=5, output_dim=1, seed=1)
    X = np.zeros((7, 3), dtype=np.float64)
    y = net.forward(X)
    assert y.shape == (7, 1)
    assert np.all(np.isfinite(y))


def test_training_decreases_loss():
    rng = np.random.RandomState(0)
    X = rng.randn(100, 1)
    y = 2.0 * X + 0.1 * rng.randn(100, 1)
    net = ThreeLayerNet(input_dim=1, hidden_dim=8, output_dim=1, seed=0)
    loss0 = net.compute_loss(X, y)
    net.fit(X, y, epochs=20, batch_size=16, learning_rate=0.1)
    loss1 = net.compute_loss(X, y)
    assert loss1 < loss0


def test_predict_reproducible_with_seed():
    net1 = ThreeLayerNet(input_dim=2, hidden_dim=4, output_dim=1, seed=5)
    net2 = ThreeLayerNet(input_dim=2, hidden_dim=4, output_dim=1, seed=5)
    assert np.allclose(net1.W1, net2.W1)
    assert np.allclose(net1.W2, net2.W2)

