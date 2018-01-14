from inspect import isclass
from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple, List, Dict
from collections import OrderedDict, namedtuple

import numpy as np
from recordclass import recordclass

from .activation import BaseActivation

Shape = Tuple[Optional[int], ...]

NpArrayTuple = Tuple[np.ndarray, ...]

ParameterInfo = namedtuple("ParameterInfo", ["shape", "init_random"])


class ShapeDescription(tuple):

    def complies(self, shape: Shape) -> bool:
        """
        Check that a shape complies with the descriptor. It will return false if not compliant
        :param shape: The shape to check for compliance
        """
        if any(map(lambda axis: axis is None, shape)):
            raise ValueError(f"Cannot validate partially defined shapes: {shape}")

        # Check size of shapes
        if len(shape) != len(self):
            return False

        # Check each dimension
        for expected_axes, axes in zip(self, shape):
            if expected_axes is not None and expected_axes != axes:
                return False  # Does not comply

        return True

    def assert_compliance(self, shape: Shape) -> None:
        """
        Assure that a shape complies with the description, otherwise raise exception
        :param shape: The shape to check for compliance
        """
        if not shape in self:
            raise TypeError(f"Expecting {self} shape but {shape} was given")

    def __contains__(self, shape: Shape) -> bool:
        """
        Shortcut operator "in" that does the same thing as complies()
        """
        return self.complies(shape)


class Layer(metaclass=ABCMeta):
    """
    Abstract definition of neural network's layer
    """

    def __init__(self, name: Optional[str] = None, depth: int = None):
        """
        Initialize layer
        :param name: The name of the layer, it should be unique in the context of model.
        If omitted it will assign one when it is attached to the model.
        :param depth: The depth of the layer. This will be updated by the model that is
        attached to.
        """
        self.name = name
        self.depth = depth
        self._input_layers = []
        self._output_layers = []

    @property
    @abstractmethod
    def parameters(self) -> OrderedDict:
        """Get a dictionary with trainable parameters of the layer."""
        pass

    @parameters.setter
    @abstractmethod
    def parameters(self, new_parameters: OrderedDict):
        """Assign new values of the trainable parameters. This process accepts a
        partial-set of parameters keep the rest unaffected."""
        pass

    @property
    @abstractmethod
    def parameters_info(self) -> Dict[str, ParameterInfo]:
        """Get information about trainable parameters"""
        pass

    @classmethod
    @abstractmethod
    def layer_type(cls) -> str:
        """A unique name of the layer type"""
        pass

    @property
    @abstractmethod
    def output_shape(self) -> ShapeDescription:
        """Get the shape of layer's output"""
        pass

    @property
    @abstractmethod
    def input_shape(self) -> ShapeDescription:
        """Get the shape of layer's input"""
        pass

    @property
    @abstractmethod
    def cached_output(self):
        # TODO: probably remove this
        pass

    @property
    def input_layers(self) -> List['Layer']:
        """Get the list with connected input layers."""
        return self._input_layers

    def __call__(self, *input_layers: 'Layer') -> 'Layer':
        """
        Connect input with other layers
        For layers that accept multiple inputs, the order of the input layers should be respected."""
        self._input_layers.extend(input_layers)
        return self

    @abstractmethod
    def forward(self, *inputs: np.ndarray) -> np.ndarray:
        """
        Perform forward propagation on this layer
        :param In: The input values of the layer
        :return: The output values of this layer
        """
        pass

    @abstractmethod
    def backward(self, dOut: np.ndarray) -> Tuple[NpArrayTuple, NpArrayTuple]:
        """
        Perform backward propagation and get the dIn/dLoss as also the gradients
        for all parameters of the model
        :param dOut: The derivative of the output of this layer according to the loss function
        :return: The dInputs derivatives for each layer along with the layers parameters gradients
        """
        pass


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
