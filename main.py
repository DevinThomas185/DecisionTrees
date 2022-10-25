from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import sys

from decision_tree import Node, DecisionTreeLeaf, DecisionTreeNode

def split_dataset(
    dataset: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    x = dataset[:, :-1]
    y = dataset[:, -1]
    return (x, y)

def read_dataset(
    filepath: str,
) -> Tuple[np.ndarray, np.ndarray]:
    dataset = np.loadtxt(filepath)
    return split_dataset(dataset)

def find_split(
    training_dataset: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    pass

def decision_tree_learning(
    training_dataset: np.ndarray, 
    depth: int
) -> Tuple[Node, int]:
    train_x, train_y = split_dataset(training_dataset)

    unique_ys = np.unique(train_y)
    if len(unique_ys) == 1:
        return DecisionTreeLeaf(str(unique_ys[0]))
    else:
        left_split, right_split = find_split(training_dataset)
        new_node: Node = None
        left_branch, left_depth = decision_tree_learning(left_split, depth + 1)
        right_branch, right_depth = decision_tree_learning(right_split, depth + 1)
        return (new_node, max(left_depth, right_depth))

if __name__ == "__main__":
    read_dataset("./intro2ML-coursework1/wifi_db/clean_dataset.txt")
    # sys.exit(read_dataset("/intro2ML-coursework1/wifi_db/noisy_dataset.txt"))

