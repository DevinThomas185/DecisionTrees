from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import sys

def read_dataset(
    filepath: str,
) -> Tuple[np.ndarray, np.ndarray]:
    x = []
    y = []
    for line in open(filepath):
        if line.strip() != "":
            row = line.strip().split("\t")
            x.append(list(map(float, row[:-1])))
            y.append(row[-1])

    return (np.array(x), np.array(y))

def decision_tree_learning(matrix: np.ndarray, depth: int):
    pass


if __name__ == "__main__":
    read_dataset("./intro2ML-coursework1/wifi_db/clean_dataset.txt")
    # sys.exit(read_dataset("/intro2ML-coursework1/wifi_db/noisy_dataset.txt"))

