import numpy as np

def get_entropy(y_dataset: np.ndarray) -> float:
    _, counts = np.unique(y_dataset, return_counts=True)
    summation:float = 0
    for frequency in counts:
        probability = frequency / float(y_dataset.size)
        summation += probability * np.log2(probability)
    
    return -summation

def get_remainder(left_split: np.ndarray, right_split: np.ndarray) -> float:
    left_remainder = left_split.size / (left_split.size + right_split.size) * get_entropy(left_split)
    right_remainder = right_split.size / (left_split.size + right_split.size) * get_entropy(right_split)
    return left_remainder + right_remainder


def get_entropy_gain(dataset: np.array, left_split: np.ndarray, right_split: np.ndarray) -> float:
    return get_entropy(dataset) - get_remainder(left_split, right_split)

def get_majority_label(y_dataset: np.array) -> float:
    labels, counts = np.unique(y_dataset, return_counts=True)

    frequency_map = dict(zip(labels, counts))
     
    highest_count = max(counts)
    for label in labels:
        if frequency_map[label] == highest_count:
            return label



