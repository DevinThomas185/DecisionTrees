from typing import Tuple
from decision_tree import Node, DecisionTreeNode, DecisionTreeLeaf
import numpy as np


def get_depth(
    tree: Node,
) -> int:
    """
    Get the depth of a tree for post pruning depth analysis

    :param tree: The tree to get the depth of
    :return: The depth of the tree
    """
    if isinstance(tree, DecisionTreeNode):
        return 1 + max(get_depth(tree.left_node), get_depth(tree.right_node))
    else:
        assert isinstance(tree, DecisionTreeLeaf)
        return 1


def evaluate_sample(
    sample: np.ndarray,
    tree: Node,
) -> Tuple[int, int]:
    """
    Return the actual class index and the predicted class index for a given
    sample

    :param sample: The sample being tested
    :param tree: The resultant decision tree to evaluate the sample on
    :return: Tuple of the real class index and the predicted class index
    """
    current_node = tree
    while isinstance(current_node, DecisionTreeNode):
        go_left = current_node.function(sample)
        if go_left:
            current_node = current_node.left_node
        else:
            current_node = current_node.right_node

    assert isinstance(current_node, DecisionTreeLeaf)
    real_class_index = int(sample[-1])
    tree_class_index = current_node.class_index
    return (real_class_index, tree_class_index)


def evaluate(
    test_dataset: np.ndarray,
    tree: Node,
) -> float:
    """
    Evaluate the accuracy of the tree on a whole test set without computing the
    confusion matrix

    :param test_dataset: The dataset being used for the test
    :param tree: The tree being tested
    :return: The accuracy of the tree on the test set
    """
    success = 0
    total = 0
    for entry in test_dataset:
        total += 1
        real_label, tree_label = evaluate_sample(entry, tree)
        if real_label == tree_label:
            success += 1

    return success / total


def get_confusion_matrix(
    test_dataset: np.ndarray,
    tree: Node,
    num_classes: int,
) -> np.ndarray:
    """
    Get the confusion matrix for a tree based on a test set

    :param test_dataset: The test set being used for the confusion matrix
    :param tree: The tree being tested
    :param num_classes: The number of unique classes possible for this set
    :return: The confusion matrix
    """
    confusion_matrix = np.zeros((num_classes, num_classes))

    for entry in test_dataset:
        real_class_index, tree_class_index = evaluate_sample(entry, tree)
        confusion_matrix[real_class_index, tree_class_index] = (
            confusion_matrix[real_class_index, tree_class_index] + 1
        )

    return confusion_matrix


def get_overall_accuracy(
    confusion_matrix: np.ndarray,
) -> float:
    """
    Get the overall accuracy from using a confusion matrix

    :param confusion_matrix: The confusion matrix
    :return: The overall accuracy
    """
    return np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)


def _get_tp_tn_fp_fn(
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


def get_f1(
    confusion_matrix: np.ndarray,
    class_index: int,
) -> float:
    """
    Get the F1 measure for a given class from the confusion matrix

    :param confusion_matrix: The confusion matrix being evaluated
    :param class_index: The class being measured
    :return: The F1 measure for this class
    """
    return get_f_beta(confusion_matrix, class_index, 1.0)


def get_f_beta(
    confusion_matrix: np.ndarray,
    class_index: int,
    beta: float = 1.0,
) -> float:
    """
    Get the F-Beta measure for a given class from the confusion matrix
    and a given beta value

    :param confusion_matrix: The confusion matrix being evaluated
    :param class_index: The class being measured
    :param beta: The beta value
    :return: The F-Beta measure for this class
    """
    precision = get_precision(confusion_matrix, class_index)
    recall = get_recall(confusion_matrix, class_index)
    beta_sq = np.power(beta, 2)
    return (1 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall)


def get_accuracy(
    confusion_matrix: np.ndarray,
    class_index: int,
) -> float:
    """
    Get the accuracy for a class given the confusion matrix

    :param confusion_matrix: The confusion matrix being evaluated
    :param class_index: The class being measured
    :return: The accuracy for this class
    """
    tp, tn, fp, fn = _get_tp_tn_fp_fn(confusion_matrix, class_index)
    return (tp + tn) / (tp + tn + fp + fn)


def get_precision(
    confusion_matrix: np.ndarray,
    class_index: int,
) -> float:
    """
    Get the precision for a class given the confusion matrix

    :param confusion_matrix: The confusion matrix being evaluated
    :param class_index: The class being measured
    :return: The precision for this class
    """
    tp, _, fp, _ = _get_tp_tn_fp_fn(confusion_matrix, class_index)
    return tp / (tp + fp)


def get_recall(
    confusion_matrix: np.ndarray,
    class_index: int,
) -> float:
    """
    Get the recall for a class given the confusion matrix

    :param confusion_matrix: The confusion matrix being evaluated
    :param class_index: The class being measured
    :return: The recall for this class
    """
    tp, _, _, fn = _get_tp_tn_fp_fn(confusion_matrix, class_index)
    return tp / (tp + fn)
