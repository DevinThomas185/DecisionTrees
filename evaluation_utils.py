from typing import List, Tuple
from decision_tree import Node, DecisionTreeNode, DecisionTreeLeaf
import numpy as np


def evaluate_sample(
    sample: np.ndarray, tree: Node, unique_classes: List[str]
) -> Tuple[str, str]:
    current_node = tree
    while isinstance(current_node, DecisionTreeNode):
        go_left = current_node.function(sample)
        if go_left:
            current_node = current_node.left_node
        else:
            current_node = current_node.right_node

    assert isinstance(current_node, DecisionTreeLeaf)
    real_label = unique_classes[int(sample[-1])]
    tree_label = current_node.get_classification()
    return (real_label, tree_label)


def evaluate(test_dataset: np.ndarray, tree: Node, unique_classes: List[str]) -> float:
    success = 0
    total = 0
    for entry in test_dataset:
        total += 1
        real_label, tree_label = evaluate_sample(entry, tree, unique_classes)
        if real_label == tree_label:
            success += 1

    return success / total


def get_confusion_matrix(
    test_dataset: np.ndarray, tree: Node, num_classes: int, unique_classes: List[str]
) -> np.ndarray:
    confusion_matrix = np.zeros((num_classes, num_classes))

    for entry in test_dataset:
        real_label, tree_label = evaluate_sample(entry, tree, unique_classes)
        r = unique_classes.index(real_label)
        t = unique_classes.index(tree_label)
        confusion_matrix[t, r] = confusion_matrix[t, r] + 1

    return confusion_matrix


def get_overall_accuracy(
    confusion_matrix: np.ndarray,
) -> float:
    return np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)


def get_tp_tn_fp_fn(
    confusion_matrix: np.ndarray, index_for_class: int
) -> Tuple[int, int, int, int]:
    tp, tn, fp, fn = 0, 0, 0, 0
    num_classes = len(confusion_matrix)
    for i in range(num_classes):
        for j in range(num_classes):
            if i == index_for_class and j == index_for_class:
                tp += confusion_matrix[i, j]
            elif i == index_for_class:
                fn += confusion_matrix[i, j]
            elif j == index_for_class:
                fp += confusion_matrix[i, j]
            elif i == j:
                tn += confusion_matrix[i, j]

    return (tp, tn, fp, fn)


def get_f1(confusion_matrix: np.ndarray, class_index: int) -> float:
    return get_f_beta(confusion_matrix, class_index, 1.0)


def get_f_beta(
    confusion_matrix: np.ndarray, class_index: int, beta: float = 1.0
) -> float:
    precision = get_precision(confusion_matrix, class_index)
    recall = get_precision(confusion_matrix, class_index)
    beta_sq = np.power(beta, 2)
    return (1 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall)


def get_accuracy(confusion_matrix: np.ndarray, class_index: int) -> float:
    tp, tn, fp, fn = get_tp_tn_fp_fn(confusion_matrix, class_index)
    return (tp + tn) / (tp + tn + fp + fn)


def get_precision(confusion_matrix: np.ndarray, class_index: int) -> float:
    tp, _, fp, _ = get_tp_tn_fp_fn(confusion_matrix, class_index)
    return tp / (tp + fp)


def get_recall(confusion_matrix: np.ndarray, class_index: int) -> float:
    tp, _, _, fn = get_tp_tn_fp_fn(confusion_matrix, class_index)
    return tp / (tp + fn)
