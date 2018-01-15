import numpy as np
from collections import OrderedDict
from typing import List, Tuple

from .base import Layer, Shape, NpArrayTuple, ShapeDescription


class Input(Layer):
    """
    The input layer that is responsible to feed the network with examples
    """

    def __init__(self, shape: Shape):
        """
        Initialize layer
        :param shape: The shape of the expected data
        """
        self._shape = ShapeDescription(shape)
        super().__init__()

    @property
    def parameters(self) -> OrderedDict:
        return OrderedDict()

    @parameters.setter
    def parameters(self, new_parameters: OrderedDict):
        raise NotImplementedError()

    @property
    def parameters_info(self):
        return {}

    @property
    def output_shape(self):
        return self._shape

    @property
    def input_shape(self) -> ShapeDescription:
        return self._shape

    @property
    def cached_output(self):
        raise NotImplementedError("Cannot be implemented for Input")

    def layer_type(cls):
        return "input"

    @property
    def input_layers(self) -> List:
        return []

    def forward(self, x_in: np.ndarray) -> np.ndarray:
        self.input_shape.assert_compliance(x_in.shape)
        return x_in

    def backward(self, dOut: np.ndarray) -> Tuple[NpArrayTuple, NpArrayTuple]:
        return (dOut,), tuple()

    def __str__(self):
        return f"{self.layer_type()}({self.input_shape})"
