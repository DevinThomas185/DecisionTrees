import os
import numpy as np

path = 'intro-to-ml-coursework-reading_dataset-intro2ML-coursework1/intro2ML-coursework1/wifi_db/clean_dataset.txt'


def read_dataset(data_set):
    data = []
    lines = 0
    for line in open(data_set):
        if line.strip() != "":
            row = line.strip().split()
            assert (
                len(row) == 8), f'Read {len(row)} row elements'
            entry = list(map(float, row[:]))
            data.append(entry)
            lines += 1

    data = np.array(data)
    print(lines, "lines read")
    return data

print(read_dataset(path))
