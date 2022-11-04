from file_utils import get_kminus1_and_1_split
import numpy as np
from typing import Tuple, List
from decision_tree import Node, DecisionTreeLeaf, DecisionTreeNode
from pruning_tree import prune_tree
from evaluation_utils import (
    get_confusion_matrix,
    evaluate,
    get_depth,
)
from math_utils import get_information_gain
from plot_tree import plot_tree
import time


def _find_split(
    dataset: np.ndarray,
    class_counts: List[int],
) -> Tuple[DecisionTreeNode, np.ndarray, List[int], np.ndarray, List[int]]:
    num_features = np.shape(dataset)[1] - 1

    best_information_gain = 0
    best_split_value = 0

    # Retain a split index so that we do not have to copy the dataset each time
    # to set the best above and below splits
    best_split_index = 0
    best_split_feature = 0

    # Retain a list representing the number of each class in each of the 'left'
    # and 'right' subtrees. Entropy calculations are much faster when dealing
    # with just counts and not the entire dataset itself
    # A previous version of the coursework used this method and the noisy dataset
    # with pruning would take about 25 inutes, now it is just 2 minutes
    best_left_class_counts = []
    best_right_class_counts = []

    # List of the sorted dataset sorted by each feature - prevents re-sorting or
    # copying the dataset to return
    sorteds = []
    for i in range(num_features):
        # We sort the dataset according to the i-th feature.
        sorted = dataset[dataset[:, i].argsort()]
        sorteds.append(sorted)

        # Required to reset the counts back to their original states when
        # looking at a new feature
        left_class_counts = [0] * len(class_counts)
        right_class_counts = class_counts.copy()

        # Split off the first sample so that the we consider legitimate split
        # points (i.e not resulting in an empty split)
        first_sample_class_index = int(sorted[0][-1])
        left_class_counts[first_sample_class_index] += 1
        right_class_counts[first_sample_class_index] -= 1

        # We iterate over the sorted array and compute the entropy gain
        # of a possible split if the classified label differs between two
        # adjacent entries.
        for j, entry in enumerate(sorted[1:]):
            class_index = entry[-1]
            feature_value = entry[i]
            previous_value = sorted[j][i]
            split_value = (feature_value + previous_value) / 2

            information_gain = get_information_gain(
                class_counts=class_counts,
                left_class_counts=left_class_counts,
                right_class_counts=right_class_counts,
            )

            if information_gain > best_information_gain:
                best_information_gain = information_gain
                best_split_index = j + 1
                best_split_feature = i
                best_split_value = split_value
                best_left_class_counts = left_class_counts.copy()
                best_right_class_counts = right_class_counts.copy()

            left_class_counts[int(class_index)] += 1
            right_class_counts[int(class_index)] -= 1

    return (
        DecisionTreeNode(
            function=lambda x: x[best_split_feature] < best_split_value,
            node_label="x[{}] < {}".format(best_split_feature, best_split_value),
        ),
        sorteds[best_split_feature][:best_split_index],
        best_left_class_counts,
        sorteds[best_split_feature][best_split_index:],
        best_right_class_counts,
    )


def _decision_tree_learning(
    dataset: np.ndarray,
    class_counts: List[str],
    unique_classes: List[str],
    depth: int = 0,
) -> Tuple[Node, int]:

    # If there is only one class with >0 count, we can create a leaf
    if len([x for x in class_counts if x != 0]) == 1:
        class_index = class_counts.index(next(filter(lambda x: x != 0, class_counts)))
        new_leaf = DecisionTreeLeaf(unique_classes[class_index], class_index)
        # The number of samples classififed in this leaf - important for knowing the
        # majority class when pruning
        new_leaf.frequency = sum(class_counts)
        return (new_leaf, depth + 1)
    # Find the optimal split point, along with the corresponding datasets for each
    # subtree and the counts for each class in them
    (
        new_node,
        left_dataset,
        left_class_counts,
        right_dataset,
        right_class_counts,
    ) = _find_split(dataset, class_counts)

    # Create the two new subtrees recursively and add them to the new node
    left_tree, left_depth = _decision_tree_learning(
        dataset=left_dataset,
        class_counts=left_class_counts,
        unique_classes=unique_classes,
        depth=depth + 1,
    )
    right_tree, right_depth = _decision_tree_learning(
        dataset=right_dataset,
        class_counts=right_class_counts,
        unique_classes=unique_classes,
        depth=depth + 1,
    )

    new_node.set_right_node(right_tree)
    new_node.set_left_node(left_tree)

    return (new_node, max(left_depth, right_depth))


def no_pruning(
    k_folds: int,
    dataset: np.ndarray,
    unique_classes: List[str],
    plot_trees: bool,
    debug: bool,
) -> np.ndarray:
    """
    Run the decision tree models without pruning

    :param k_folds: The number of folds to apply to the cross validation
    :param dataset: The training dataset
    :param unique_classes: List of the strings for each class possible
    :param plot_trees: Flag to turn on or off the image generation of trees
    :param debug: Flag to turn on debugging prints
    :return: The resulting confusion matrix after training
    """

    num_classes = len(unique_classes)
    overall_confusion_matrix = np.zeros((num_classes, num_classes))

    # Create a generator that returns a new train, test split
    train_test_split = get_kminus1_and_1_split(dataset, k_folds)

    for i in range(k_folds):
        start = time.time()
        if debug:
            print("\nTest Fold:", i)

        # Get the next train, test splits
        train, test = next(train_test_split)
        class_counts = np.unique(train[:, -1], return_counts=True)[1].tolist()

        tree, depth = _decision_tree_learning(train, class_counts, unique_classes)
        end = time.time()

        if plot_trees:
            plot_tree(tree, "tree{}.svg".format(i))

        if debug:
            acc = evaluate(test, tree)
            print("Accuracy: {:.3f} \tTime: {:.3f}".format(acc, end - start))

        overall_confusion_matrix += get_confusion_matrix(test, tree, num_classes)

    return overall_confusion_matrix


def pruning(
    k_folds: int,
    dataset: np.ndarray,
    unique_classes: List[str],
    plot_trees: bool,
    debug: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the decision tree models with pruning

    :param k_folds: The number of folds to apply to the cross validation
    :param dataset: The training dataset
    :param unique_classes: List of the strings for each class possible
    :param plot_trees: Flag to turn on or off the image generation of trees
    :param debug: Flag to turn on debugging prints
    :return: The resulting confusion matrices before and after pruning
    """

    num_classes = len(unique_classes)
    overall_confusion_matrix = np.zeros((num_classes, num_classes))
    before_confusion_matrix = np.zeros((num_classes, num_classes))
    train_test_generator = get_kminus1_and_1_split(dataset, k_folds)

    # For depth analysis of pruning
    total_depth_before = 0
    total_depth_after = 0

    # We use a generator to return to us the test and training sets to use for
    # a given iteration
    for i in range(k_folds):
        temp_train, test = next(train_test_generator)
        train_validation_generator = get_kminus1_and_1_split(temp_train, k_folds - 1)

        # Another generator is used to iterate over the splits for the validation
        # and training datasets
        for j in range(k_folds - 1):
            start = time.time()
            train, validation = next(train_validation_generator)
            if debug:
                print("\nTest Fold: {} \tValidation Fold: {}".format(i, j))

            class_counts = np.unique(train[:, -1], return_counts=True)[1].tolist()

            tree, depth = _decision_tree_learning(
                dataset=train,
                class_counts=class_counts,
                unique_classes=unique_classes,
            )
            total_depth_before += depth

            # For results analysis before and after pruning, we accumulate
            # the confusion matrix before pruning
            before_confusion_matrix += get_confusion_matrix(test, tree, num_classes)

            if debug:
                prev_acc = evaluate(validation, tree)

            if plot_trees:
                plot_tree(tree, "{}_{}_before.svg".format(i, j))

            # Prune the tree - edits in place
            prune_tree(
                root=tree,
                current_node=tree,
                validation=validation,
            )
            end = time.time()

            if plot_trees:
                plot_tree(tree, "{}_{}_after.svg".format(i, j))

            if debug:
                new_acc = evaluate(validation, tree)
                total_depth_after += get_depth(tree)
                print(
                    "Accuracy: {:.3f} -> {:.3f} \tTime: {:.3f}".format(
                        prev_acc, new_acc, end - start
                    )
                )

            # Accumulate the confusion matrix after pruning
            overall_confusion_matrix += get_confusion_matrix(test, tree, num_classes)

    if debug:
        print(
            "\nAverage Depth Before: {}".format(
                total_depth_before / (k_folds * (k_folds - 1))
            )
        )
        print(
            "Average Depth After: {}".format(
                total_depth_after / (k_folds * (k_folds - 1))
            )
        )
    # Return the overall confusion matrix before and after pruning for analysis
    # of performance improvement from pruning
    return overall_confusion_matrix, before_confusion_matrix
