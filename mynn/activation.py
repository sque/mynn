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

    def __repr__(self) -> str:
        """
        Just return the name of the final class
        """
        class_name = self.__class__.__name__.split('.')[-1]
        end = class_name.find("Activation")
        return class_name[:end]


class SigmoidActivation(BaseActivation):
    """
    Implementation of Sigmoid activation function
    """

    def __call__(self, Z: FloatOrArray) -> FloatOrArray:
        return 1 / (1 + np.exp(-Z))

    def derivative(self, Z: FloatOrArray) -> FloatOrArray:
        return self(Z) * (1 - self(Z))


class SoftmaxActivation(BaseActivation):
    """
    Implementation of softmax activation function
    """

    def __call__(self, Z: FloatOrArray) -> FloatOrArray:

        t = np.exp(Z - np.max(Z, axis=0))   # Normalize with a constant
        return t / np.sum(t, axis=0, keepdims=True)

    def derivative(self, Z: FloatOrArray) -> FloatOrArray:
        # Computing the derivative of softmax is not that easy...
        # Also it is more computentially efficient to use dZ = AL - Y when combining
        # Singmoid/Softmax with CrossEntropy loss function
        raise NotImplementedError()

        # For reference this was an attempt to implement jacobian
        # J = - Z[..., None] * Z[:, None, :]  # off-diagonal Jacobian
        # iy, ix = np.diag_indices_from(J[0])
        # J[:, iy, ix] = Z * (1. - Z)  # diagonal
        # return J.sum(axis=1)  # sum across-rows for each sample


class TanhActivation(BaseActivation):
    """
    Implementation of Tanh activation function
    """

    def __call__(self, Z: FloatOrArray) -> FloatOrArray:
        return np.tanh(Z)

    def derivative(self, Z: FloatOrArray) -> FloatOrArray:
        return 1 - np.tanh(Z) ** 2


class ReLUActivation(BaseActivation):
    """
    Implementation of ReLU activation function
    """

    def __call__(self, Z: FloatOrArray) -> FloatOrArray:
        return np.maximum(0, Z)

    def derivative(self, Z: FloatOrArray) -> FloatOrArray:
        if isinstance(Z, (float, np.float)):
            return int(Z > 0)
        else:
            return np.array(Z > 0, dtype='float')
