"""Utility helpers: minibatches, losses, weight init, simple checks."""
import numpy as np
from typing import Iterator, Tuple, Optional


def iter_minibatches(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True, rng: Optional[np.random.RandomState] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (X_batch, y_batch) for one epoch.
    If batch_size >= n_samples, yields one batch of full data.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = X.shape[0]
    if rng is None:
        rng = np.random.RandomState()
    indices = np.arange(n)
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    diff = y_true - y_pred
    return float(np.mean(np.square(diff)))


def init_weights(shape: Tuple[int, int], method: str = "xavier", rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Initialize weight matrix of shape (in_dim, out_dim).
    Supported methods: 'xavier' (default), 'he', 'normal'.
    """
    if rng is None:
        rng = np.random.RandomState()
    fan_in, fan_out = shape
    if method == "xavier":
        # Glorot normal
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return rng.normal(loc=0.0, scale=std, size=shape).astype(np.float64)
    elif method == "he":
        std = np.sqrt(2.0 / fan_in)
        return rng.normal(loc=0.0, scale=std, size=shape).astype(np.float64)
    else:
        return rng.normal(loc=0.0, scale=0.01, size=shape).astype(np.float64)


def assert_2d_array(x: np.ndarray, name: str = "array") -> None:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {x.shape}")

