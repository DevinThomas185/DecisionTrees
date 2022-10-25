import os
import numpy as np
from numpy.random import default_rng

path = 'intro-to-ml-coursework-reading_dataset-intro2ML-coursework1/intro2ML-coursework1/wifi_db/clean_dataset.txt'


def read_dataset(file):
    data = []
    data = np.loadtxt(file, dtype="float", delimiter="\t")
    return data

# Reads np-array and shuffles the data into subsets
def split_dataset(data, subsets, random_generator=default_rng()):
    shuffled_indices = random_generator.permutation(len(data))
    data_rand = np.asarray(np.array_split(data[shuffled_indices], subsets))
    return data_rand


# Example: Reads dataset and shuffles it into 6 subsets
print(split_dataset(read_dataset(path), 6))
