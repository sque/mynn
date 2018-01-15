from typing import Optional, Dict, Tuple
from inspect import isclass
from collections import OrderedDict

import numpy as np
from recordclass import recordclass

from mynn.activation import BaseActivation
from .base import Layer, ShapeDescription, Shape, ParameterInfo, NpArrayTuple


class FullyConnected(Layer):
    """
    Fully connected layer implements the typical dense linear network with
    optional non-linear activation.
    """

    LayerParametersType = recordclass('LayerParameters', ['W', 'b'])
    ForwardCacheType = recordclass('ForwardCacheType', ['In', 'Z', 'Out', 'extras'])

    def __init__(self, units: int, activation: Optional[BaseActivation] = None, name: Optional[str] = None) -> None:
        """
        Initialize layer
        :param units: The number of units in this layer
        :param activation: The activation function
        :param name: The name of the layer.
        """
        self._units = units
        self._cache = self.ForwardCacheType(None, None, None, {})
        self._parameters = self.LayerParametersType(W=None, b=None)

        super().__init__(name=name)

        self._activation: BaseActivation = activation
        if isclass(self._activation):
            # If a class is give, instantiate it.
            self._activation = self._activation()

    def layer_type(cls):
        return "FC"

    @property
    def units(self) -> int:
        """Get the number of units"""
        return self._units

    @property
    def parameters(self):
        return self._parameters._asdict()

    @parameters.setter
    def parameters(self, new_parameters: OrderedDict):
        for k, v in new_parameters.items():
            self.parameters_info[k].shape.assert_compliance(v.shape)
            setattr(self._parameters, k, v)

    @property
    def output_shape(self) -> Shape:
        return (self._units, None)

    @property
    def input_shape(self) -> Shape:
        return self.input_layers[0].output_shape

    @property
    def parameters_info(self) -> Dict[str, ParameterInfo]:
        return {
            'W': ParameterInfo(shape=ShapeDescription((self._units, self.input_shape[0])), init_random=True),
            'b': ParameterInfo(shape=ShapeDescription((self._units, 1)), init_random=False)
        }

    @property
    def cached_output(self):
        return self._cache.Out

    def forward(self, In: np.ndarray) -> np.ndarray:
        self._cache.In = In

        # Calculate linear output
        self._cache.Z = np.dot(self._parameters.W, In) + self._parameters.b

        # Calculate activation function
        if self._activation:
            self._cache.Out = self._activation(self._cache.Z)
        else:
            self._cache.Out = self._cache.Z
        return self._cache.Out

    def backward(self, dOut: np.ndarray) -> Tuple[NpArrayTuple, NpArrayTuple]:

        if self._activation:
            dZ = dOut * self._activation.derivative(self._cache.Z)
        else:
            dZ = dOut

        dW = np.dot(dZ, self._cache.In.T) / self._cache.In.shape[-1]
        db = np.mean(dZ, axis=1, keepdims=True)
        dIn = np.dot(self._parameters.W.T, dZ)

        return (dIn,), (dW, db)

    def __str__(self):
        if self._activation:
            activation = f"╭╯ {self._activation}"
        else:
            activation = ""
        return f"{self.layer_type()}({self.units}) {activation} "
