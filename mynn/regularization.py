from typing import Union, List
import numpy as np
from .value_types import LayerValues, LayerGrads, LayerParameters


class RegularizationBase:
    """
    Base class for implementing regularization methods
    """

    def on_post_forward_propagation(self, layer_values: LayerValues,
                                    layer_index: int,
                                    layer_params: LayerParameters) -> LayerValues:
        """
        Callback to be called after forward propagation.
        :param layer_values: The layer values as calculated by forward propagation
        :param layer_index: The index of the current layer
        :param layer_params: The parameters of the current layer
        :return: The altered layer_values
        """
        return layer_values

    def on_pre_backward_propagation(self, dZ: np.ndarray,
                                    layer_index: int,
                                    samples: int,
                                    layer_values: LayerValues,
                                    layer_params: LayerParameters) -> LayerValues:
        """
        Callback to be called before backward propagation.
        :param dZ: The derivative of loss function against dZ
        :param layer_index: The index of the current layer
        :param layer_params: The parameters of the current layer
        :return: The altered dA
        """
        return dZ

    def on_post_backward_propagation(self, grads: LayerGrads, layer_index: int,
                                     samples: int,
                                     layer_values: LayerValues,
                                     layer_params: LayerParameters) -> LayerGrads:
        """
        Callback to be called after backward propagation.
        :param grads: The calculated gradients
        :param layer_index: The index of the current layer
        :param layer_params: The parameters of the current layer
        :return: The altered gradients
        """
        return grads


class L2Regularization(RegularizationBase):
    """
    Implementation of L2 regularization technique.
    """

    def __init__(self, lambd: float):
        """
        Initialize L2 regularizer
        :param lambd: The λ term of the regularization
        """
        self._lambd = lambd

    def on_post_backward_propagation(self, grads: LayerGrads, layer_index: int,
                                     samples: int,
                                     layer_values: LayerValues,
                                     layer_params: LayerParameters) -> LayerGrads:
        l2_regularization_term = self._lambd * layer_params.W / (2 * samples)
        return LayerGrads(dW=grads.dW + l2_regularization_term, db=grads.db)

    def __repr__(self):
        return f"{self.__class__.__name__}(λ={self._lambd})"


class DropoutRegularization(RegularizationBase):
    """
    Implementation of inverse dropout regularization technique.
    """

    def __init__(self, keep_probs: Union[float, List[float]]):
        """
        Initialize regularizer
        :param keep_probs: The probability of a node to not been dropped out per layer. If
        one float is given the the same probability will be used for all layers.
        """
        if not isinstance(keep_probs, (list, float)):
            raise TypeError(f"Unexpected type {type(keep_probs)} for keep_probs. Expecting float or list of floats")
        self._keep_probs = keep_probs

    def keep_proba(self, layer_index: int) -> float:
        """
        Get keep probability for a specific layer index
        :param layer_index: The index of the layer (1-based) that requesting keep probability
        :returns The keep probability per neuron
        """
        if isinstance(self._keep_probs, float):
            return self._keep_probs
        return self._keep_probs[layer_index - 1]

    def on_post_forward_propagation(self, layer_values: LayerValues,
                                    layer_index: int,
                                    layer_params: LayerParameters) -> LayerValues:
        keep_proba = self.keep_proba(layer_index=layer_index)
        layer_values.extras['dropout_keep_proba'] = keep_proba
        if keep_proba == 1:
            # Disabled drop-out for this layer
            return layer_values

        dropout_mask = np.random.rand(*layer_values.A.shape)
        dropout_mask = np.array(dropout_mask < keep_proba, dtype='float') / keep_proba
        A = (layer_values.A * dropout_mask)
        layer_values.extras['dropout'] = dropout_mask
        return LayerValues(Z=layer_values.Z, A=A, extras=layer_values.extras)

    def on_pre_backward_propagation(self, dZ: np.ndarray,
                                    layer_index: int,
                                    samples: int,
                                    layer_values: LayerValues,
                                    layer_params: LayerParameters) -> LayerValues:
        keep_proba = self.keep_proba(layer_index=layer_index)
        if keep_proba == 1:
            # Disabled drop-out for this layer
            return dZ
        dropout_mask = layer_values.extras['dropout']
        dZ = dZ * dropout_mask
        return dZ

    def __repr__(self):
        return f"{self.__class__.__name__}(keep_probs={self._keep_probs})"
