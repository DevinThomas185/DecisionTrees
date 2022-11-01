import numpy as np
from typing import List, Optional, Tuple


def read_dataset(
    filepath: str,
) -> Tuple[np.ndarray, List[str]]:
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
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def get_kminus1_and_1_split(
    dataset: np.ndarray,
    k_folds: int,
) -> Tuple[np.ndarray, np.ndarray]:
    splits = np.split(dataset, k_folds)

    for i in range(k_folds):
        if i == 0:
            train = splits[i + 1 :]
        elif i == k_folds - 1:
            train = splits[:i]
        else:
            train = np.concatenate((splits[:i], splits[i + 1 :]), axis=0)

        train = np.concatenate(train, axis=0)
        test = splits[i]
        yield (train, test)
