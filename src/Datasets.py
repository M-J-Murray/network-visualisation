import numpy as np
from enum import Enum


class Dataset(Enum):
    CIRCLE = "circle",
    SPIRAL = "spiral",
    NESTED = "nested_circle",
    COMPLEX = "complex_circle"


def perfume_dataset(file_path):
    X = np.zeros((560, 2))
    y = np.zeros((560), dtype='uint8')
    per_class = 28
    sel = np.arange(per_class*2)
    with open(file_path, "r") as file:
        current_label = 0
        for line in file:
            c_i = current_label * per_class
            csv_split = np.array(line.split(","))
            X[c_i:(c_i+per_class), 0] = np.array(csv_split[sel[sel % 2 == 0] + 1], dtype="int")
            X[c_i:(c_i+per_class), 1] = np.array(csv_split[sel[sel % 2 == 1] + 1], dtype="int")
            y[c_i:(c_i+per_class)] = current_label
            current_label += 1
    return X, y


def spiral_dataset(points, classes):
    N = points  # number of points per class
    D = 2  # dimensionality
    K = classes  # number of classes
    X = np.zeros((N * K, D))  # data matrix (each row = single example)
    y = np.zeros(N * K, dtype='uint8')  # class labels
    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    return X, y


def circle_dataset(samples: int, min_range: float, max_range: float, radius: float):
    range_ = max_range - min_range
    X = (np.random.rand(samples, 2) * range_) + min_range
    y = np.array(np.sqrt(np.sum(np.multiply(X, X), axis=1)) > radius, dtype="uint8")
    return X, y


def nested_circle_dataset(samples: int, min_range: float, max_range: float, radius1: float, radius2: float, order=(1, 2)):
    range_ = max_range - min_range
    X = (np.random.rand(samples, 2) * range_) + min_range
    y = np.zeros(samples, dtype="uint8")
    y[np.sqrt(np.sum(np.multiply(X, X), axis=1)) < radius1] = order[0]
    y[np.sqrt(np.sum(np.multiply(X, X), axis=1)) < radius2] = order[1]
    return X, y


def complex_circles_dataset(samples: int, min_range: float, max_range: float, radius1: float, radius2: float):
    offset = (max_range - min_range) / 4
    c1_X, c1_y = nested_circle_dataset(int(samples / 4), min_range / 2, max_range / 2, radius1, radius2, order=(2, 1))
    c2_X, c2_y = nested_circle_dataset(int(samples / 4), min_range / 2, max_range / 2, radius1, radius2)
    c3_X, c3_y = nested_circle_dataset(int(samples / 4), min_range / 2, max_range / 2, radius1, radius2)
    c4_X, c4_y = nested_circle_dataset(int(samples / 4), min_range / 2, max_range / 2, radius1, radius2, order=(2, 1))

    c1_X += np.array((-offset, offset))
    c2_X += np.array((offset, offset))
    c3_X += np.array((-offset, -offset))
    c4_X += np.array((offset, -offset))

    return np.concatenate((c1_X, c2_X, c3_X, c4_X), axis=0), np.concatenate((c1_y, c2_y, c3_y, c4_y), axis=0)
