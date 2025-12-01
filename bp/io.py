"""Simple parameter save/load helpers using numpy.savez / load."""
import numpy as np
from typing import Dict


def save_parameters(path: str, params: Dict[str, np.ndarray]) -> None:
    np.savez(path, **params)


def load_parameters(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}

