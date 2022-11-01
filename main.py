from argparse import ArgumentParser
import sys
from typing import Dict, Generator, List, Optional, Tuple, Set
import numpy as np
import file_utils
import math_utils
import evaluation_utils
import time

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
    training_dataset: np.ndarray, used_splits: Set[Tuple[int, float]]
) -> Tuple[DecisionTreeNode, np.ndarray, np.ndarray, Set[Tuple[int, float]]]:
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

            # Update the current label
            current_label = label

            # Read in the feature values of the two entries that we might
            # want to split between.
            feature_value = entry[i]
            previous_value = sorted[j][i]
            split_value = (feature_value + previous_value) / 2

            # Heuristics for skipping splitting points
            """
            # If the class of two entries is the same, we don't consider
            # splitting between them.
            if label == current_label:
                continue

            # If the feature values are the same, skip
            if feature_value == previous_value:
                continue
            """

            # If we have already split on some value for a given i-th feature
            # we don't want to split again on that.
            if (i, split_value) in used_splits:
                pass

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
    used_splits.add((best_split_feature, best_split_value))
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
    unique_classes: List[str],
    depth: int = 0,
    split_features: Set[Tuple[int, float]] = set(),
) -> Tuple[Node, int]:
    _, y_train = get_features_and_labels(training_dataset)

    unique_ys = np.unique(y_train)
    if len(unique_ys) == 1:
        new_leaf = DecisionTreeLeaf(unique_classes[int(unique_ys[0])])
        new_leaf.frequency = len(training_dataset)
        return (new_leaf, depth + 1)
    else:
        new_node, left_split, right_split, new_used_splits = find_split(
            training_dataset, split_features
        )
        left_branch, left_depth = decision_tree_learning(
            left_split, unique_classes, depth + 1, new_used_splits
        )
        right_branch, right_depth = decision_tree_learning(
            right_split, unique_classes, depth + 1, new_used_splits
        )
        new_node.set_left_node(left_branch)
        new_node.set_right_node(right_branch)
        return (new_node, max(left_depth, right_depth))


def prune_tree(
    root: Node,
    current_node: Node,
    validation: np.ndarray,
    unique_classes: List[str],
) -> Node:
    if isinstance(current_node, DecisionTreeLeaf):
        return

    assert isinstance(current_node, DecisionTreeNode)
    prune_tree(root, current_node.left_node, validation, unique_classes)
    prune_tree(root, current_node.right_node, validation, unique_classes)

    if isinstance(current_node.left_node, DecisionTreeLeaf) and isinstance(
        current_node.right_node, DecisionTreeLeaf
    ):
        old_classification_error = 1 - evaluation_utils.evaluate(
            validation, root, unique_classes
        )

        is_left = current_node.parent.left_node == current_node

        new_leaf_class = (
            current_node.left_node.get_classification()
            if current_node.left_node.frequency > current_node.right_node.frequency
            else current_node.right_node.get_classification()
        )
        new_leaf = DecisionTreeLeaf(new_leaf_class)
        new_leaf.frequency = (
            current_node.left_node.frequency + current_node.right_node.frequency
        )

        if is_left:
            current_node.parent.set_left_node(new_leaf)
        else:
            current_node.parent.set_right_node(new_leaf)

        new_classification_error = 1 - evaluation_utils.evaluate(
            validation, root, unique_classes
        )

        # Undo since tree is now worse
        if new_classification_error > old_classification_error:
            if is_left:
                current_node.parent.set_left_node(current_node)
            else:
                current_node.parent.set_right_node(current_node)


def no_pruning(
    unique_classes: List[str],
    train_test_split: Generator[Tuple[np.ndarray, np.ndarray], None, None],
    plot_trees: bool,
    k_folds: int,
    debug: bool,
) -> np.ndarray:

    num_classes = len(unique_classes)
    overall_confusion_matrix = np.zeros((num_classes, num_classes))

    for i in range(k_folds):
        start = time.time()
        if debug:
            print("\nIteration", i)

        train, test = next(train_test_split)
        tree, depth = decision_tree_learning(train, unique_classes)

        end = time.time()

        if plot_trees:
            plot_tree(tree, depth, "tree{}.svg".format(i))

        if debug:
            acc = evaluation_utils.evaluate(test, tree, unique_classes)
            print("Accuracy: {:.3f} \tTime: {:.3f}".format(acc, end - start))

        confusion_matrix = evaluation_utils.get_confusion_matrix(
            test, tree, 4, unique_classes
        )

        overall_confusion_matrix += confusion_matrix

    return overall_confusion_matrix


def pruning(
    unique_classes: List[str],
    train_test_split: Generator[Tuple[np.ndarray, np.ndarray], None, None],
    plot_trees: bool,
    k_folds: int,
    debug: bool,
) -> np.ndarray:

    num_classes = len(unique_classes)
    overall_confusion_matrix = np.zeros((num_classes, num_classes))

    for i in range(k_folds):
        old_train, test = next(train_test_split)
        for j in range(k_folds - 1):
            start = time.time()
            validation_train_split = file_utils.get_kminus1_and_1_split(
                old_train, k_folds - 1
            )
            train, validation = next(validation_train_split)

            if debug:
                print("\nIteration", i, j)

            tree, depth = decision_tree_learning(train, unique_classes)

            # Prune tree
            if debug:
                prev_acc = evaluation_utils.evaluate(validation, tree, unique_classes)
            if plot_trees:
                plot_tree(tree, depth, "before_{}_{}.svg".format(i, j))

            prune_tree(tree, tree, validation, unique_classes)

            if plot_trees:
                plot_tree(tree, depth, "after_{}_{}.svg".format(i, j))

            if debug:
                new_acc = evaluation_utils.evaluate(validation, tree, unique_classes)
                print(
                    "Accuracy: {:.3f} -> {:.3f} \tTime: {:.3f}".format(
                        prev_acc, new_acc, end - start
                    )
                )

            end = time.time()
            confusion_matrix = evaluation_utils.get_confusion_matrix(
                test, tree, num_classes
            )

            overall_confusion_matrix += confusion_matrix

    return overall_confusion_matrix


def run_decision_tree(
    k_folds: int,
    path_to_dataset: str,
    with_pruning: bool,
    plot_trees: bool,
    debug: bool,
    seed: Optional[int],
) -> None:
    start = time.time()

    dataset, unique_classes = file_utils.read_dataset(path_to_dataset)
    shuffled_dataset = file_utils.shuffle_dataset(dataset, seed=seed)
    num_classes = len(unique_classes)

    train_test_split = file_utils.get_kminus1_and_1_split(shuffled_dataset, k_folds)

    if with_pruning:
        overall_confusion_matrix = pruning(
            unique_classes=unique_classes,
            train_test_split=train_test_split,
            plot_trees=plot_trees,
            k_folds=k_folds,
            debug=debug,
        )
    else:
        overall_confusion_matrix = no_pruning(
            unique_classes=unique_classes,
            train_test_split=train_test_split,
            plot_trees=plot_trees,
            k_folds=k_folds,
            debug=debug,
        )

    for i in range(num_classes):
        print()
        print(
            "Class {} Accuracy: {}".format(
                i, evaluation_utils.get_accuracy(overall_confusion_matrix, i)
            )
        )
        print(
            "Class {} Precision: {}".format(
                i, evaluation_utils.get_precision(overall_confusion_matrix, i)
            )
        )
        print(
            "Class {} Recall: {}".format(
                i, evaluation_utils.get_recall(overall_confusion_matrix, i)
            )
        )
        print(
            "Class {} F1: {}".format(
                i, evaluation_utils.get_f1(overall_confusion_matrix, i)
            )
        )

    print(
        "\nOverall Accuracy: {}".format(
            evaluation_utils.get_overall_accuracy(overall_confusion_matrix)
        )
    )

    end = time.time()

    if debug:
        print("\nResulting Overall Confusion Matrix:")
        print(overall_confusion_matrix)
        print("\nTotal Time Taken: {:.3f}".format(end - start))


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Decision Tree", description="Decision Tree and Evaluation Metrics"
    )

    parser.add_argument(
        "PATH_TO_DATASET",
        help="The relative path to the dataset",
    )

    parser.add_argument(
        "--visualise",
        "-v",
        action="store_true",
        help="Use this flag to produce images visualising the decision trees",
    )

    parser.add_argument(
        "--k_folds",
        "-k",
        help="The number of folds to use to split the dataset up into",
        default=10,
        type=int,
    )

    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Use this flag to turn on tree pruning utilising a validation dataset fold",
    )

    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Use this flag to turn on debugging prints",
    )

    parser.add_argument(
        "-s",
        "--seed",
        help="Provide a seed for shuffling the dataset, leave empty for random seed",
        default=None,
        type=int,
    )

    args = parser.parse_args(sys.argv[1:])

    run_decision_tree(
        k_folds=args.k_folds,
        path_to_dataset=args.PATH_TO_DATASET,
        with_pruning=args.pruning,
        plot_trees=args.visualise,
        debug=args.debug,
        seed=args.seed,
    )

    # "./intro2ML-coursework1/wifi_db/clean_dataset.txt"
    # "./intro2ML-coursework1/wifi_db/noisy_dataset.txt"
