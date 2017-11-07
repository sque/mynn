import numpy as np

from ._const import SMALL_FLOAT, FloatOrArray


class BaseLossFunction:
    """
    Base class for implementing base loss functions
    """

    def __call__(self, A: FloatOrArray, Y: FloatOrArray) -> FloatOrArray:
        """
        Given two vectors calculate the loss function
        :param A: The real outcome of the neural network
        :param Y: The expected output of the neural network
        :return: The loss of the network
        """
        raise NotImplementedError()

    def derivative(self, A: FloatOrArray) -> FloatOrArray:
        """
        Calculate the derivative of the function for the value A
        :param A: The values to calculate derivatives
        :return: The derivative of the number or the array in the same
        shape as A
        """
        raise NotImplementedError()


class CrossEntropyLoss:

    def __call__(self, A: FloatOrArray, Y: FloatOrArray) -> FloatOrArray:
        """
        Given two vectors calculate the loss function
        :param A: The real outcome of the neural network
        :param Y: The expected output of the neural network
        :return: The loss of the network
        """
        A = np.where(A == 0, SMALL_FLOAT, A)  # Replace all zero with a very small number
        logprobs = np.multiply(np.log(A), Y) + np.multiply((1 - Y), np.log((1 - A)))
        return np.squeeze(- np.mean(logprobs))

    def derivative(self, A: FloatOrArray, Y: FloatOrArray) -> FloatOrArray:
        """
        Calculate the derivative of the function for the value A
        :param A: The values to calculate derivatives
        :return: The derivative of the number or the array in the same
        shape as A
        """
        return - (np.divide(Y, A) - np.divide(1 - Y, 1 - A))
