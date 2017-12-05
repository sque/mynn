from typing import Union, List, Tuple
import numpy as np


class WeightInitializerBase:
    pass

    def get_initial_weight(self, l: int, n: int, n_left: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the initial weights of the linear functions Z
        :param l: The level of the layer 1-indexed
        :param n: The number of nodes in the layer (Noutput)
        :param n_left: The number of nodes in the previous layer (Ninput)
        :return: An array of shape (n, n_left) with the initial values of linear weights as also the b weights
        """
        raise NotImplementedError()


class ConstantWeightInitializer(WeightInitializerBase):
    """
    It initializes all weights with constant values.
    """

    def __init__(self, weights: List[Union[np.array, float]]):
        """
        Create a constant weight initializer
        :param weights: A List with constant weights per layer. For each layer you can give
        a correct shaped array or a float to be broadcast.
        """
        self._weights = weights

    def get_initial_weight(self, l: int, n: int, n_left: int) -> Tuple[np.ndarray, np.ndarray]:
        layer_weights = self._weights[l - 1]
        b = np.zeros((n, 1))

        if isinstance(layer_weights, (float, int)):
            return np.ones((n, n_left)) * layer_weights, b
        elif isinstance(layer_weights, np.ndarray):
            if layer_weights.shape != (n, n_left):
                raise ValueError(f"The provided weight array is not of the correct shape for layer {l}")
            return layer_weights, b

        raise TypeError(f"Unexpected type of: {type(layer_weights)} for layer {l}")


class NormalWeightInitializer(WeightInitializerBase):
    """
    Weight initializer with normal distribution
    """

    def get_initial_weight(self, l: int, n: int, n_left: int) -> Tuple[np.ndarray, np.ndarray]:
        W = np.random.randn(n, n_left)
        b = np.zeros((n, 1))
        return W, b


class VarianceScalingWeightInitializer(WeightInitializerBase):
    """
    A general form of an initializer that tries to scale the variance of the weights

    """

    def __init__(self, scale: float = 2):
        """
        Weights will be initialized as W = randn() * np.sqrt(scale / n_left)
        :param scale: The scale factor of the final variance
        """
        self._scale = scale

    def get_initial_weight(self, l: int, n: int, n_left: int) -> Tuple[np.ndarray, np.ndarray]:
        W = np.random.randn(n, n_left) * np.sqrt(self._scale / n_left)
        b = np.zeros((n, 1))
        return W, b

    def __repr__(self):
        return f"{self.__class__.__name__}({self._scale})"