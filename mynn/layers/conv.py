import numpy as np
from typing import Optional, Dict, Tuple
from inspect import isclass
from collections import OrderedDict

from recordclass import recordclass

from mynn.activation import BaseActivation
from .base import Layer, ShapeDescription, Shape, ParameterInfo, NpArrayTuple


class Conv2D(Layer):
    """2D Convolutional layer"""

    LayerParametersType = recordclass('LayerParameters', ['KW', 'Kb'])
    ForwardCacheType = recordclass('ForwardCacheType', ['In', 'Z', 'Out'])

    def __init__(self, filters: int, kernel_shape: Shape, stride: int = 1, padding: str = 'same',
                 activation: Optional[BaseActivation] = None, name: Optional[str] = None):
        super().__init__(name=name)

        # Check arguments
        if padding not in {'same', 'valid'}:
            raise ValueError(f"Unknown type of padding \"padding\". Expecting \"same\" or \"valid\"")

        self._kernel_shape = (kernel_shape, kernel_shape) if isinstance(kernel_shape, int) else kernel_shape
        self._padding_type = padding or 'same'
        self._filters = filters
        self._parameters = self.LayerParametersType(KW=None, Kb=None)
        self._cache = self.ForwardCacheType(In=None, Z=None, Out=None)
        self._stride = stride
        self._padding_size = (0,0)
        self._output_shape = None

        self._activation: BaseActivation = activation
        if isclass(self._activation):
            # If a class is give, instantiate it.
            self._activation = self._activation()

    def _calculate_padding_size(self):
        if self._padding_type == 'valid':
            return (0, 0, 0, 0)

    def layer_type(cls):
        return "Conv2D"

    @property
    def padding_type(self) -> str:
        """The type of padding as "same" or "valid" """
        return self._padding_type

    @property
    def parameters(self):
        return self._parameters._asdict()

    @parameters.setter
    def parameters(self, new_parameters: OrderedDict):
        for k, v in new_parameters.items():
            self.parameters_info[k].shape.assert_compliance(v.shape)
            setattr(self._parameters, k, v)

    @property
    def parameters_info(self) -> Dict[str, ParameterInfo]:
        return {
            'KW': ParameterInfo(
                shape=ShapeDescription(self._kernel_shape + (self.input_shape[2], self._filters)),
                init_random=True),
            'Kb': ParameterInfo(
                shape=ShapeDescription((1, 1, 1, self._filters)),
                init_random=False)
        }

    @property
    def input_shape(self) -> Shape:
        return self.input_layers[0].output_shape

    @property
    def output_shape(self) -> Shape:
        return self._output_shape

    def __call__(self, *args, **kwargs):
        result = super().__call__(*args, **kwargs)

        if self._padding_type == 'same':
            # Calculate padding size
            self._padding_size = (
                int(self._kernel_shape[0] / 2),
                int(self._kernel_shape[0] / 2)
            )

        self._output_shape = ShapeDescription((
            int((self.input_shape[0] + 2 * self._padding_size[0] - self._kernel_shape[0]) / self._stride) + 1,
            int((self.input_shape[0] + 2 * self._padding_size[1] - self._kernel_shape[0]) / self._stride) + 1,
            self._filters,
            None)
        )

        return result

    @property
    def cached_output(self):
        raise NotImplementedError()

    def forward(self, In: np.ndarray) -> np.ndarray:

        # Pad input image if needed
        if any(self._padding_size):
            h_pad, w_pad = self._padding_size
            In = np.pad(In,
                        (
                            (h_pad, h_pad),
                            (w_pad, w_pad),
                            (0, 0), (0, 0)
                        ),
                        mode='constant',
                        constant_values=0)

        self._cache.In = In

        # Initialize empty array of convolution output
        # Format of H, W, C, M
        self._cache.Z = np.empty(self.output_shape[:-1] + (In.shape[-1], ))

        for m in range(In.shape[-1]):
            for f in range(self._filters):
                for h_out in range(self.output_shape[0]):
                    for w_out in range(self.output_shape[1]):

                        # Calculate source coordinates
                        top = h_out * self._stride
                        left = w_out * self._stride
                        bottom = top + self._kernel_shape[0]
                        right = left + self._kernel_shape[1]

                        # Calculate convoluted pixel
                        convoluted = np.sum(In[top:bottom, left:right,:,m] * self._parameters.KW[:,:,:,f])
                        convoluted += float(self._parameters.Kb.reshape(-1)[f])

                        # Save to the output
                        self._cache.Z[h_out, w_out, f, m] = convoluted

        if self._activation:
           self._cache.Out = self._activation(self._cache.Z)
        else:
            self._cache.Out = self._cache.Z

        return self._cache.Out


    def backward(self, dOut: np.ndarray) -> Tuple[NpArrayTuple, NpArrayTuple]:
        raise NotImplementedError()

    def __str__(self):
        if self._activation:
            activation = f"╭╯ {self._activation}"
        else:
            activation = ""
        return f"{self.layer_type()}({self._kernel_shape[0]}x{self._kernel_shape[1]}) {activation} "
