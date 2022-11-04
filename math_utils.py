import numpy as np
from typing import List


def _get_entropy(
    class_counts: List[int],
) -> float:
    total: float = 0
    total_samples = sum(class_counts)
    for frequency in class_counts:
        if frequency == 0:
            continue

        probability = frequency / float(total_samples)
        total += probability * np.log2(probability)
    return -total


def _get_remainder(
    left_class_counts: List[int],
    right_class_counts: List[int],
) -> float:
    total_left = sum(left_class_counts)
    total_right = sum(right_class_counts)
    total_all_samples = total_right + total_left
    left_remainder = total_left / total_all_samples * _get_entropy(left_class_counts)
    right_remainder = total_right / total_all_samples * _get_entropy(right_class_counts)
    return left_remainder + right_remainder


def get_information_gain(
    class_counts: List[int],
    left_class_counts: List[int],
    right_class_counts: List[int],
) -> float:
    """
    Get the information gain value for a partition of a dataset

    :param class_counts: List of counts for each class for the whole dataset
    :param left_class_counts: List of counts for each class for the left dataset
    :param right_class_counts: List of counts for each class for the right dataset
    :return: The information gain from this partition
    """
    return _get_entropy(class_counts) - _get_remainder(
        left_class_counts, right_class_counts
    )
