from argparse import ArgumentParser
import sys
from decision_tree_learning import pruning, no_pruning
from file_utils import read_dataset, shuffle_dataset
from typing import Optional
import time
from evaluation_utils import (
    get_accuracy,
    get_precision,
    get_recall,
    get_f1,
    get_overall_accuracy,
)


def _run_decision_tree(
    k_folds: int,
    path_to_dataset: str,
    with_pruning: bool,
    plot_trees: bool,
    debug: bool,
    seed: Optional[int],
) -> None:
    """
    Run the decision tree models and print the relevant evaluation metrics

    :param k_folds: The number of folds to apply to the cross validation
    :param path_to_dataset: The path to the dataset
    :param with_pruning: Flag to turn on or off pruning of the tree
    :param plot_trees: Flag to turn on or off the image generation of trees
    :param debug: Flag to turn on debugging prints
    :param seed: Optional seed for shuffling
    """
    start = time.time()

    # Read and shuffle the dataset
    dataset, unique_classes = read_dataset(path_to_dataset)
    shuffled_dataset = shuffle_dataset(dataset=dataset, seed=seed)
    num_classes = len(unique_classes)

    if with_pruning:
        overall_confusion_matrix, overall_confusion_matrix_before = pruning(
            k_folds=k_folds,
            dataset=shuffled_dataset,
            unique_classes=unique_classes,
            plot_trees=plot_trees,
            debug=debug,
        )
    else:
        overall_confusion_matrix = no_pruning(
            k_folds=k_folds,
            dataset=shuffled_dataset,
            unique_classes=unique_classes,
            plot_trees=plot_trees,
            debug=debug,
        )
        pass

    for i in range(num_classes):
        print()
        print(
            "Class {} Accuracy: {}".format(i, get_accuracy(overall_confusion_matrix, i))
        )
        print(
            "Class {} Precision: {}".format(
                i, get_precision(overall_confusion_matrix, i)
            )
        )
        print("Class {} Recall: {}".format(i, get_recall(overall_confusion_matrix, i)))
        print("Class {} F1: {}".format(i, get_f1(overall_confusion_matrix, i)))

    if with_pruning:
        before_accuracy = get_overall_accuracy(overall_confusion_matrix_before)
        after_accuracy = get_overall_accuracy(overall_confusion_matrix)
        print("\nOverall Accuracy Before Pruning: {}".format(before_accuracy))
        print("Overall Accuracy After Pruning: {}".format(after_accuracy))
        print("Improvement in Accuracy: {}".format(after_accuracy - before_accuracy))
    else:
        print(
            "\nOverall Accuracy: {}".format(
                get_overall_accuracy(overall_confusion_matrix)
            )
        )

    end = time.time()

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

    _run_decision_tree(
        k_folds=args.k_folds,
        path_to_dataset=args.PATH_TO_DATASET,
        with_pruning=args.pruning,
        plot_trees=args.visualise,
        debug=args.debug,
        seed=args.seed,
    )
