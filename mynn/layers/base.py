from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple, List, Dict
from collections import OrderedDict, namedtuple

import numpy as np

# Typing types
Shape = Tuple[Optional[int], ...]
NpArrayTuple = Tuple[np.ndarray, ...]

# Parameter Info type
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
        self._input_layers: List[Layer] = []

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
        Connect input of this with output of other layers
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
