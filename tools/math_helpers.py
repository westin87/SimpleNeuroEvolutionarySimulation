import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relative_difference_of(first: int, second: int) -> float:
    return np.abs(first - second) / (np.max([first, second]) + np.finfo(np.float32).eps)