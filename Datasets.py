import numpy as np
from enum import Enum


class Dataset(Enum):
    CIRCLE = "circle",
    SPIRAL = "spiral"


def get_circle_dataset(points: int, min_range: float, max_range: float, radius: float):
    # generating labelled training data
    range_ = max_range-min_range
    N = points
    X = (np.random.rand(N, 2) * range_) + min_range
    y = np.sqrt(np.sum(np.multiply(X, X), axis=1)) > radius
    return np.stack((X[:, 0], X[:, 1], y), axis=1)


def get_spiral_dataset(points: int, classes: int):
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
    return np.stack((X[:, 0], X[:, 1], y), axis=1)
