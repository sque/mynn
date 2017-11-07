import numpy as np

from ._const import FloatOrArray


class BaseActivation:
    """
    Base class for implementing activation functions
    """

    def __call__(self, Z: FloatOrArray) -> FloatOrArray:
        """
        Calculate the direct result of function for Z value
        :param Z: An array of values to calculate the result
        of the functions
        :return: A number or an array of values in the same shape
        as Z.
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


class SigmoidActivation(BaseActivation):
    """
    Implementation of Sigmoid activation function
    """

    def __call__(self, Z: FloatOrArray) -> FloatOrArray:
        return 1 / (1 + np.exp(-Z))

    def derivative(self, A: FloatOrArray) -> FloatOrArray:
        return self(A) * (1 - self(A))


class TanhActivation(BaseActivation):
    """
    Implementation of Tanh activation function
    """

    def __call__(self, Z: FloatOrArray) -> FloatOrArray:
        return np.tanh(Z)

    def derivative(self, A: FloatOrArray) -> FloatOrArray:
        return 1 - A ** 2


class ReLUActivation(BaseActivation):
    """
    Implementation of ReLU activation function
    """

    def __call__(self, Z: FloatOrArray) -> FloatOrArray:
        return np.maximum(0, Z)

    def derivative(self, A: FloatOrArray) -> FloatOrArray:
        return np.array(A > 0, dtype='float')
