"""Example script to train ThreeLayerNet on simple function approximation tasks."""
import os
import argparse
import numpy as np
from bp.network import ThreeLayerNet


def make_dataset(task: str, n_samples: int = 200, seed: int = 0):
    rng = np.random.RandomState(seed)
    if task == "sin":
        X = rng.uniform(-np.pi, np.pi, size=(n_samples, 1))
        y = np.sin(X)
    elif task == "square":
        X = rng.uniform(-1.0, 1.0, size=(n_samples, 1))
        y = X ** 2
    else:
        raise ValueError("Unknown task")
    return X.astype(np.float64), y.astype(np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["sin", "square"], default="sin")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X, y = make_dataset(args.task, n_samples=500, seed=args.seed)
    # train/test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    net = ThreeLayerNet(input_dim=1, hidden_dim=50, output_dim=1, seed=args.seed)
    print("Starting training...")
    net.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch, learning_rate=args.lr, verbose=1)
    train_mse = net.evaluate(X_train, y_train)
    test_mse = net.evaluate(X_test, y_test)
    print(f"Train MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}")

    os.makedirs("models", exist_ok=True)
    fname = f"models/three_layer_{args.task}.npz"
    net.save(fname)
    print(f"Saved model to {fname}")


if __name__ == "__main__":
    main()

