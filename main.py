from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
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
    training_dataset: np.ndarray,
    used_splits: List[Tuple[int, float]]
) -> DecisionTreeNode:
    '''
    Here we need to sort the values for each feature and consider split points 
    which are between two different class labels
    '''
    # Sort the dataset for each feature [x0, x1, x2, x3, x4, x5, x6]
    num_features = np.shape(training_dataset)[1] - 1
    _, y_dataset_all = get_features_and_labels(training_dataset)
    split_feature = 0
    split_value = 0
    best_above_split = np.array([])
    best_below_split = np.array([])
    best_entropy_gain = 0

    for i in range(num_features):
        sorted = training_dataset[training_dataset[:, i].argsort()]
        # Find split points per sorted dataset (split point is where the class_label changes)
        above_split = np.array([sorted[0]])
        below_split = np.copy(sorted[1:])
        current_label = sorted[0][-1]
        for j, entry in enumerate(sorted[1:]):
            feature_value = entry[i]
            label = entry[-1]
            if label != current_label:
                if (i, feature_value) in used_splits:
                    above_split = np.append(
                        arr=above_split, values=[entry], axis=0)
                    below_split = np.delete(arr=below_split, obj=0, axis=0)
                    continue

                current_label = label
                # Compute information gain here
                _, y_dataset_above = get_features_and_labels(above_split)
                _, y_dataset_below = get_features_and_labels(below_split)
                entropy_gain = math_utils.get_entropy_gain(
                    y_dataset_all, y_dataset_above, y_dataset_below)
                if entropy_gain > best_entropy_gain:
                    best_entropy_gain = entropy_gain
                    best_above_split = np.copy(above_split)
                    best_below_split = np.copy(below_split)
                    split_feature = i
                    split_value = (feature_value + sorted[j][i]) / 2

            above_split = np.append(arr=above_split, values=[entry], axis=0)
            below_split = np.delete(arr=below_split, obj=0, axis=0)

    used_splits.append((split_feature, split_value))
    return (DecisionTreeNode(function=lambda x: x[split_feature] < split_value,
                             node_label="x[{}] < {}".format(split_feature, split_value)), best_above_split, best_below_split, used_splits)


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
        new_node, left_split, right_split, new_split_features = find_split(
            training_dataset, split_features)
        left_branch, left_depth = decision_tree_learning(
            left_split, depth + 1, new_split_features)
        right_branch, right_depth = decision_tree_learning(
            right_split, depth + 1, new_split_features)
        new_node.set_left_node(left_branch)
        new_node.set_right_node(right_branch)
        return (new_node, max(left_depth, right_depth))


def test_accuracy(
    test_dataset: np.ndarray,
    tree: Node
) -> float:
    success = 0
    total = 0
    for entry in test_dataset:
        total += 1
        current_node = tree
        while isinstance(current_node, DecisionTreeNode):
            go_left = current_node.function(entry)
            if go_left:
                current_node = current_node.left_node
            else:
                current_node = current_node.right_node

        assert (isinstance(current_node, DecisionTreeLeaf))
        real_label = str(entry[-1])
        tree_label = current_node.classification
        if real_label == tree_label:
            success += 1

    return success / total


if __name__ == "__main__":
    dataset = file_utils.read_dataset(
        "./intro2ML-coursework1/wifi_db/clean_dataset.txt")
    train, test, validation = file_utils.split_dataset(dataset, [0.6, 0.8])
    tree, depth = decision_tree_learning(train, 0, [])
    tree.print_tree(0)
    print(test_accuracy(test, tree))
    print(test_accuracy(train, tree))
    print(test_accuracy(validation, tree))

    # a = DecisionTreeLeaf('a')
    # b = DecisionTreeLeaf('b')
    # c = DecisionTreeLeaf('c')
    # d = DecisionTreeLeaf('d')
    # e = DecisionTreeNode(lambda x: x, 'e')
    # f = DecisionTreeNode(lambda x: x, 'f')
    # g = DecisionTreeNode(lambda x: x, 'g')
    # e.set_left_node(a)
    # e.set_right_node(b)
    # f.set_left_node(c)
    # f.set_right_node(d)
    # g.set_left_node(e)
    # g.set_right_node(f)

    plot_tree(tree, depth, "tree.svg")
