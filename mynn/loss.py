import numpy as np

from ._const import SMALL_FLOAT


def cross_entropy_loss(A:np.ndarray, Y:np.ndarray) -> np.ndarray:
    """
    Given two vectors calculate the cross entropy loss
    :param A: The real outcome of the neural network
    :param Y: The expected output of the neural network
    :return: The loss of the network
    """
    A = np.where(A == 0, SMALL_FLOAT, A) # Replace all zero with a very small number
    logprobs = np.multiply(np.log(A), Y) + np.multiply((1 - Y), np.log((1 - A)))
    return np.squeeze(- np.mean(logprobs))

