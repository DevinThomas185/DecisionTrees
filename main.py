from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import sys

def read_dataset(
    filepath: str,
) -> Tuple[np.ndarray, np.ndarray]:
    all = np.loadtxt(filepath)
    x = all[:, :-1]
    y = all[:, -1]
    return (x, y)

def decision_tree_learning(matrix: np.ndarray, depth: int):
    pass


if __name__ == "__main__":
    read_dataset("./intro2ML-coursework1/wifi_db/clean_dataset.txt")
    # sys.exit(read_dataset("/intro2ML-coursework1/wifi_db/noisy_dataset.txt"))

