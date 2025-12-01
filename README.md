Three-layer BP neural network (pure NumPy)

This project implements a simple three-layer MLP trained with backpropagation (sigmoid hidden units, linear output) using NumPy.

Quick start

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run tests:

```
python -m pytest -q
```

3. Train example:

```
python scripts/train_example.py --task sin --epochs 200
```

Files
- `bp/` package contains core implementation.
- `scripts/train_example.py` trains the network on simple regression tasks.
- `tests/` contains pytest unit tests.

