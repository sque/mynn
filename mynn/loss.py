import numpy as np
from .activation import SigmoidActivation, SoftmaxActivation
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

    def derivative(self, A: FloatOrArray, Y: FloatOrArray) -> FloatOrArray:
        """
        Calculate the derivative of the function for the value A
        :param A: The values to calculate derivatives
        :return: The derivative of the number or the array in the same
        shape as A
        """
        raise NotImplementedError()


class BinaryCrossEntropyLoss:
    """
    Specialized cross entropy for binary problems using one output unit where
    the only the P(C=1|X) is given and the P(C=0|X) is inferred by 1 - P(C=1|X)
    """

    @staticmethod
    def _clip_activations(A: FloatOrArray) -> FloatOrArray:
        """
        Clip activation values to be in open range of (0, 1)
        :param A: The outcome of the activation function
        :return: Clipped values with a very small amount
        """
        return np.clip(A, SMALL_FLOAT, 1 - SMALL_FLOAT)

    def __call__(self, A: FloatOrArray, Y: FloatOrArray) -> FloatOrArray:
        """
        Given two vectors calculate the loss function
        :param A: The real outcome of the neural network
        :param Y: The expected output of the neural network
        :return: The loss of the network
        """
        A = self._clip_activations(A)
        logprobs = np.multiply(np.log(A), Y) + np.multiply((1 - Y), np.log((1 - A)))
        return np.squeeze(- np.mean(logprobs))

    def derivative(self, A: FloatOrArray, Y: FloatOrArray) -> FloatOrArray:
        """
        Calculate the derivative of the function for the value A
        :param A: The values to calculate derivatives
        :return: The derivative of the number or the array in the same
        shape as A
        """
        A = self._clip_activations(A)
        return - (np.divide(Y, A) - np.divide(1 - Y, 1 - A))


class CrossEntropyLoss:
    """
    General cross-entropy
    """

    @staticmethod
    def _clip_activations(A: FloatOrArray) -> FloatOrArray:
        """
        Clip activation values to be in open range of (0, 1)
        :param A: The outcome of the activation function
        :return: Clipped values with a very small amount
        """
        return np.clip(A, SMALL_FLOAT, 1 - SMALL_FLOAT)

    def __call__(self, A: FloatOrArray, Y: FloatOrArray) -> FloatOrArray:
        """
        Given two vectors calculate the loss function
        :param A: The real outcome of the neural network
        :param Y: The expected output of the neural network
        :return: The loss of the network
        """
        A = self._clip_activations(A)
        logprobs = np.sum(np.multiply(np.log(A), Y), axis=0)
        return np.squeeze(- np.mean(logprobs))

    def derivative(self, A: FloatOrArray, Y: FloatOrArray) -> FloatOrArray:
        """
        Calculate the derivative of the function for the value A
        :param A: The values to calculate derivatives
        :return: The derivative of the number or the array in the same
        shape as A
        """
        A = self._clip_activations(A)
        return - (np.divide(Y, A) - np.divide(1 - Y, 1 - A))


class SigmoidCrossEntropyLoss(BinaryCrossEntropyLoss):
    """
    Cross entropy with sigmoid response
    """

    _activation = SigmoidActivation()

    def __call__(self, A: FloatOrArray, Y: FloatOrArray) -> FloatOrArray:
        """
        Given two vectors calculate the loss function
        :param A: The real outcome of the neural network
        :param Y: The expected output of the neural network
        :return: The loss of the network
        """
        return super().__call__(self._activation(A), Y)

    def derivative(self, A: FloatOrArray, Y: FloatOrArray) -> FloatOrArray:
        """
        Calculate the derivative of the function for the value A
        :param A: The values to calculate derivatives
        :return: The derivative of the number or the array in the same
        shape as A
        """
        return self._activation(A) - Y


class SoftmaxCrossEntropyLoss(CrossEntropyLoss):
    """
    Cross entropy with softmax response
    """

    _activation = SoftmaxActivation()

    def __call__(self, A: FloatOrArray, Y: FloatOrArray) -> FloatOrArray:
        """
        Given two vectors calculate the loss function
        :param A: The real outcome of the neural network
        :param Y: The expected output of the neural network
        :return: The loss of the network
        """
        return super().__call__(self._activation(A), Y)

    def derivative(self, A: FloatOrArray, Y: FloatOrArray) -> FloatOrArray:
        """
        Calculate the derivative of the function for the value A
        :param A: The values to calculate derivatives
        :return: The derivative of the number or the array in the same
        shape as A
        """
        return self._activation(A) - Y
