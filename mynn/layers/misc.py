from typing import Dict, Tuple
from collections import OrderedDict

import numpy as np

from .base import Layer, ShapeDescription, Shape, ParameterInfo, NpArrayTuple


class Flatten(Layer):
    """Flatten multi-dimensional input layer"""

    @property
    def parameters(self):
        return OrderedDict()

    @parameters.setter
    def parameters(self, new_parameters: OrderedDict):
        raise RuntimeError("Flatten layer does not have any trainable parameters to assign")

    @property
    def parameters_info(self) -> Dict[str, ParameterInfo]:
        return OrderedDict()

    @property
    def input_shape(self) -> Shape:
        return self.input_layers[0].output_shape

    @property
    def output_shape(self) -> Shape:
        return ShapeDescription((np.prod(self.input_shape[:-1]), None))

    def layer_type(cls):
        return "Flatten"

    @property
    def cached_output(self):
        raise NotImplementedError()

    def forward(self, In: np.ndarray) -> np.ndarray:
        return In.reshape(self.output_shape[:-1] + (In.shape[-1],))

    def backward(self, dOut: np.ndarray) -> Tuple[NpArrayTuple, NpArrayTuple]:
        return (dOut.reshape(self.input_shape[:-1] + (dOut.shape[-1],)), ), tuple()

    def __str__(self):
        return f"{self.layer_type()}()"
