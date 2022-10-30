from typing import List, Tuple
import numpy as np
import file_utils
import math_utils

from decision_tree import Node, DecisionTreeLeaf, DecisionTreeNode
from plot_tree import plot_tree

TRAINING_FRACTION = 0.8


def get_features_and_labels(
    dataset: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    x = dataset[:, :-1]
    y = dataset[:, -1]
    return (x, y)


def find_split(
    training_dataset: np.ndarray, used_splits: List[Tuple[int, float]]
) -> Tuple[DecisionTreeNode, np.ndarray, np.ndarray, List[Tuple[int, float]]]:
    num_features = np.shape(training_dataset)[1] - 1

    best_entropy_gain = 0
    best_split_value = 0
    best_split_feature = 0
    best_above_split = np.array([])
    best_below_split = np.array([])

    for i in range(num_features):
        # We sort the dataset according to the i-th feature.
        sorted = training_dataset[training_dataset[:, i].argsort()]

        # We iterate over the sorted array and compute the entropy gain
        # of a possible split if the classified label differs between two
        # adjacent entries.
        current_label = sorted[0][-1]
        for j, entry in enumerate(sorted[1:]):
            label = entry[-1]

            # If the class of two entries is the same, we don't consider
            # splitting between them.
            if label == current_label:
                continue

            # Update the current label
            current_label = label

            # Read in the feature values of the two entries that we might
            # want to split between.
            feature_value = entry[i]
            previous_value = sorted[j][i]
            split_value = (feature_value + previous_value) / 2

            # If we have already split on some value for a given i-th feature
            # we don't want to split again on that.
            if (i, split_value) in used_splits:
                continue

            above_split, below_split = split_at(split_value, i, sorted)

            entropy_gain = math_utils.get_entropy_gain(
                training_dataset, above_split, below_split
            )

            if entropy_gain > best_entropy_gain:
                best_entropy_gain = entropy_gain
                best_above_split = np.copy(above_split)
                best_below_split = np.copy(below_split)
                best_split_feature = i
                best_split_value = split_value

    # Record the best split so that we don't use it twice.
    used_splits.append((best_split_feature, best_split_value))

    return (
        DecisionTreeNode(
            function=lambda x: x[best_split_feature] < best_split_value,
            node_label="x[{}] < {}".format(best_split_feature, best_split_value),
        ),
        best_above_split,
        best_below_split,
        used_splits,
    )


def split_at(
    split_value: float, feature_index: int, dataset: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    above_split = dataset[dataset[:, feature_index] < split_value]
    below_split = dataset[dataset[:, feature_index] >= split_value]
    return (above_split, below_split)


def decision_tree_learning(
    training_dataset: np.ndarray,
    depth: int,
    split_features: List[Tuple[int, float]],
) -> Tuple[Node, int]:
    _, y_train = get_features_and_labels(training_dataset)

    unique_ys = np.unique(y_train)
    if len(unique_ys) == 1:
        return (DecisionTreeLeaf(str(unique_ys[0])), depth + 1)
    else:
        new_node, left_split, right_split, new_used_splits = find_split(
            training_dataset, split_features
        )
        left_branch, left_depth = decision_tree_learning(
            left_split, depth + 1, new_used_splits
        )
        right_branch, right_depth = decision_tree_learning(
            right_split, depth + 1, new_used_splits
        )
        new_node.set_left_node(left_branch)
        new_node.set_right_node(right_branch)
        return (new_node, max(left_depth, right_depth))


def test_accuracy(test_dataset: np.ndarray, tree: Node) -> float:
    success = 0
    total = 0
    for entry in test_dataset:
        total += 1
        real_label = str(entry[-1])
        tree_label = classify(entry, tree)
        if real_label == tree_label:
            success += 1

    return success / total


def classify(entry: np.ndarray, tree: Node) -> str:
    current_node = tree
    while isinstance(current_node, DecisionTreeNode):
        go_left = current_node.function(entry)
        if go_left:
            current_node = current_node.left_node
        else:
            current_node = current_node.right_node

    assert isinstance(current_node, DecisionTreeLeaf)
    return current_node.get_classification()


if __name__ == "__main__":
    dataset = file_utils.read_dataset(
        "./intro2ML-coursework1/wifi_db/clean_dataset.txt"
    )
    train, test, validation = file_utils.split_dataset(dataset, [0.6, 0.8])
    tree, depth = decision_tree_learning(train, 0, [])
    tree.print_tree(0)
    print(test_accuracy(test, tree))
    print(test_accuracy(train, tree))
    print(test_accuracy(validation, tree))

    plot_tree(tree, depth, "tree.svg")
