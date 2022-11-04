import numpy as np
from typing import List, Optional, Tuple


def read_dataset(
    filepath: str,
) -> Tuple[np.ndarray, List[str]]:
    """
    Read the dataset and return the transformed dataset and the list of
    the unique classes

    :param filepath: The path to the dataset
    :return: The transformed dataset and the list of the unique classes
    """
    dataset = np.loadtxt(filepath)
    features = dataset[:, :-1]
    classes = dataset[:, -1]

    unique_classes = []
    for c in classes:
        c_str = str(c)
        if c_str not in unique_classes:
            unique_classes.append(c_str)

    transformed_classes = np.array([unique_classes.index(str(c)) for c in classes])
    return (np.column_stack((features, transformed_classes)), unique_classes)


def shuffle_dataset(
    dataset: np.ndarray,
    seed: Optional[int],
) -> np.ndarray:
    """
    Shuffle the dataset according to a given seed

    :param dataset: The dataset being shuffled
    :param seed: The seed used to shuffle
    :return: The shuffled dataset
    """
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def get_kminus1_and_1_split(
    dataset: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get a split of k-1 and 1 folds based on k

    :param dataset: The dataset being broken up
    :param k: The number of sections to split the data into
    :return: The k-1 and 1 splits
    """
    splits = np.split(dataset, k)

    for i in range(k):
        if i == 0:
            k_minus_1_split = splits[i + 1 :]
        elif i == k - 1:
            k_minus_1_split = splits[:i]
        else:
            k_minus_1_split = np.concatenate((splits[:i], splits[i + 1 :]), axis=0)

        k_minus_1_split = np.concatenate(k_minus_1_split, axis=0)
        one_split = splits[i]
        yield (k_minus_1_split, one_split)
