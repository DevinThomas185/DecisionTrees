import os
import numpy as np
from typing import List

def read_dataset(
    filepath: str,
) -> np.ndarray:
    return np.loadtxt(filepath)

# Reads np-array and shuffles the data into subsets
def split_dataset(
    data, 
    split_points, 
) -> List[np.ndarray]:
    np.random.shuffle(data)
    length = len(data)
    divs = [int(sp * length) for sp in split_points]
    return np.split(data, divs)