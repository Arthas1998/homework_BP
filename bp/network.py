"""Three-layer MLP with sigmoid hidden activation and mean-squared-error loss.
Model (input -> hidden -> output) with separate forward/backward logic.
"""
import numpy as np
from typing import Optional, List, Dict
from .activations import sigmoid, sigmoid_derivative, linear, linear_derivative
from .utils import init_weights, mean_squared_error, iter_minibatches
from .io import save_parameters, load_parameters


class ThreeLayerNet:
    """Simple three-layer network (input -> hidden -> output).

    Public attributes:
      W1, b1, W2, b2 : parameters
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, activation: str = "sigmoid", weight_init: str = "xavier", dtype: np.dtype = np.float64, seed: Optional[int] = None):
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.dtype = dtype
        self.rng = np.random.RandomState(seed)

        # initialize weights
        self.W1 = init_weights((self.input_dim, self.hidden_dim), method=weight_init, rng=self.rng)
        self.b1 = np.zeros((1, self.hidden_dim), dtype=self.dtype)
        self.W2 = init_weights((self.hidden_dim, self.output_dim), method=weight_init, rng=self.rng)
        self.b2 = np.zeros((1, self.output_dim), dtype=self.dtype)

        # activation functions
        if activation == "sigmoid":
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        else:
            # fallback to sigmoid
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative

    def forward(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=self.dtype)
        # ensure 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self._X = X
        # hidden
        self._z1 = X.dot(self.W1) + self.b1  # shape (n, hidden)
        self._a1 = self.activation(self._z1)
        # output (linear)
        self._z2 = self._a1.dot(self.W2) + self.b2  # shape (n, out)
        self._out = linear(self._z2)
        return self._out

    predict = forward

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.forward(X)
        return mean_squared_error(y, y_pred)

    def _backward(self, X: np.ndarray, y: np.ndarray, outputs: np.ndarray) -> Dict[str, np.ndarray]:
        # X shape (batch, input_dim); y shape (batch, output_dim)
        n = X.shape[0]
        y = np.asarray(y, dtype=self.dtype)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        # dLoss/dout for MSE: 2*(out - y)/n
        d_out = (2.0 * (outputs - y)) / n  # shape (batch, out)
        # gradients for W2, b2
        dW2 = self._a1.T.dot(d_out)  # (hidden, out)
        db2 = np.sum(d_out, axis=0, keepdims=True)
        # backprop into hidden
        da1 = d_out.dot(self.W2.T)  # (batch, hidden)
        dz1 = da1 * self.activation_derivative(self._z1)  # elementwise
        dW1 = X.T.dot(dz1)  # (in, hidden)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.01, momentum: float = 0.0, shuffle: bool = True, verbose: int = 0) -> List[float]:
        X = np.asarray(X, dtype=self.dtype)
        y = np.asarray(y, dtype=self.dtype)
        n = X.shape[0]
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        epochs = int(epochs)
        batch_size = int(batch_size)
        # velocity for momentum
        vW1 = np.zeros_like(self.W1)
        vb1 = np.zeros_like(self.b1)
        vW2 = np.zeros_like(self.W2)
        vb2 = np.zeros_like(self.b2)

        losses: List[float] = []
        for epoch in range(epochs):
            # epoch loss
            epoch_losses = []
            for Xb, yb in iter_minibatches(X, y, batch_size=batch_size if batch_size > 0 else n, shuffle=shuffle, rng=self.rng):
                out = self.forward(Xb)
                loss = mean_squared_error(yb, out)
                epoch_losses.append(loss)
                grads = self._backward(Xb, yb, out)
                # SGD update
                if momentum and momentum != 0.0:
                    vW1 = momentum * vW1 - learning_rate * grads["dW1"]
                    vb1 = momentum * vb1 - learning_rate * grads["db1"]
                    vW2 = momentum * vW2 - learning_rate * grads["dW2"]
                    vb2 = momentum * vb2 - learning_rate * grads["db2"]
                    self.W1 += vW1
                    self.b1 += vb1
                    self.W2 += vW2
                    self.b2 += vb2
                else:
                    self.W1 -= learning_rate * grads["dW1"]
                    self.b1 -= learning_rate * grads["db1"]
                    self.W2 -= learning_rate * grads["dW2"]
                    self.b2 -= learning_rate * grads["db2"]
            mean_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else float(0.0)
            losses.append(mean_epoch_loss)
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs} - loss: {mean_epoch_loss:.6f}")
        return losses

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.compute_loss(X, y)

    def save(self, path: str) -> None:
        params = {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2,
                  "input_dim": np.array(self.input_dim), "hidden_dim": np.array(self.hidden_dim), "output_dim": np.array(self.output_dim)}
        save_parameters(path, params)

    @classmethod
    def load(cls, path: str) -> "ThreeLayerNet":
        data = load_parameters(path)
        input_dim = int(data["input_dim"]) if "input_dim" in data else int(data["W1"].shape[0])
        hidden_dim = int(data["hidden_dim"]) if "hidden_dim" in data else int(data["W1"].shape[1])
        output_dim = int(data["output_dim"]) if "output_dim" in data else int(data["W2"].shape[1])
        net = cls(input_dim, hidden_dim, output_dim)
        net.W1 = data["W1"].astype(np.float64)
        net.b1 = data["b1"].astype(np.float64)
        net.W2 = data["W2"].astype(np.float64)
        net.b2 = data["b2"].astype(np.float64)
        return net

