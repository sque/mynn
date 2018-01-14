from abc import ABCMeta, abstractmethod
from typing import Union, Dict, Tuple
from .layers import Layer
import logging as _logging
import numpy as np

logger = _logging.getLogger(__name__)


class WeightInitializerBase(metaclass=ABCMeta):
    """Base class for implementing initializer of layer's trainable parameters"""

    @abstractmethod
    def __call__(self, layer: Layer):
        """
        Initialize trainable parameters of a layer, usually called weights
        :param layer: The layer to initialize weights
        """
        pass


class ConstantWeightInitializer(WeightInitializerBase):
    """
    It initializes all weights with constant values.
    """

    def __init__(self, weights: Dict[str, Dict[str, Union[np.array, float]]]):
        """
        Create a constant weight initializer
        :param weights: Weights mapped per layer and parameter name. The weights can
        be given as a full array or as a float that will be broadcasted to the shape
        of the array.
        """
        self._weights = weights

    def __call__(self, layer: Layer):

        if not layer.name in self._weights:
            logger.warning(f"Couldn't find constant weights for layer {layer.name}. Skipping")
            return

        layer_weights = self._weights[layer.name]
        for pname, info in layer.parameters_info.items():
            if pname not in layer_weights:
                logger.warning(f"Couldn't find constant weights for parameter {pname} of "
                               f"layer {layer.name}. Skipping")
                continue
            param_weights = layer_weights[pname]
            if not isinstance(param_weights, np.ndarray):
                param_weights = np.ones(info.shape) * param_weights
            layer.parameters = {pname: param_weights}


class NormalWeightInitializer(WeightInitializerBase):
    """
    Weight initializer with normal distribution
    """

    def __init__(self, scale: float):
        """
        :param scale: The maximum values of weights.
        """
        self._scale = scale

    def __call__(self, layer: Layer):

        for p, info in layer.parameters_info.items():
            if info.init_random:
                layer.parameters = {p: np.random.rand(*info.shape) * self._scale}
            else:
                layer.parameters = {p: np.zeros(info.shape)}

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class XavierWeightInitializer(WeightInitializerBase):
    """
    A general form of an initializer that tries to scale the variance of the weights
    """

    def __init__(self, scale: float = 2):
        """
        Weights will be initialized as W = randn() * np.sqrt(scale / n_in + n_out)
        :param scale: The scale factor of the final variance
        """
        self._scale = scale

    def __call__(self, layer: Layer):

        for p, info in layer.parameters_info.items():
            if info.init_random:
                layer.parameters = {
                    p: np.random.randn(*info.shape) * np.sqrt(
                        self._scale / (layer.input_shape[0] + layer.output_shape[0]))
                }
            else:
                layer.parameters = {p: np.zeros(info.shape)}

    def __repr__(self):
        return f"{self.__class__.__name__}({self._scale})"
