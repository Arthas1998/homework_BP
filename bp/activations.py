"""Activation functions and derivatives (NumPy).
Includes sigmoid (default), tanh, relu, and linear.
"""
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically-stable sigmoid.
    Returns elementwise 1 / (1 + exp(-x))."""
    x = np.asarray(x)
    # Clip for stability might be unnecessary with float64 but keep a safe range
    # Use expit-like stable form
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid. Accepts pre-activation or activated values.
    If input looks like probabilities (in [0,1]) it's treated as sigmoid(x).
    """
    x = np.asarray(x)
    # If values are in (0,1) it's probably already sigmoid outputs
    if np.all((x >= 0.0) & (x <= 1.0)):
        s = x
    else:
        s = sigmoid(x)
    return s * (1.0 - s)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    t = np.tanh(x)
    return 1.0 - t * t


def relu(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return np.where(x > 0, x, 0.0)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return (x > 0).astype(np.float64)


def linear(x: np.ndarray) -> np.ndarray:
    return np.asarray(x)


def linear_derivative(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x, dtype=np.float64)

