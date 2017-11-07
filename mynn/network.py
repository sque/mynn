from typing import List, Optional, Tuple
from operator import itemgetter

import numpy as np

from .optimizers import OptimizerBase, AdaptiveGradientDescentMomentum
from .activation import BaseActivation, SigmoidActivation, TanhActivation
from .loss import cross_entropy_loss, BaseLossFunction, CrossEntropyLoss


class NeuralNetwork:
    """
    L-NeuralNetwork with sigmoid output and cross-entropy loss function
    """

    def __init__(self, layers_config: List[Tuple[int, BaseActivation]],
                 n_x: int,
                 init_random_weight=0.001,
                 prediction_proba_threshold: int = 0.5,
                 optimizer: Optional[OptimizerBase] = None,
                 cost_function: Optional[BaseLossFunction] = None):
        """
        Initiate an untrained neural network
        :param layers_config: A list of tuples describing each layer starting from the first hidden till the output
        layer. The tuple must consist of the number of nodes and the class of the activation function to use.
        :param n_x: Number of input features
        :param init_random_weight: Weight of the random factor to initialize model parameters
        :param prediction_proba_threshold: The probability threshold to select one class or another.
        :param optimizer: The optimizer object to use for optimization. If not defined it will use the Adaptive GD
        with default parameters.
        :param cost_function:
        """

        # Hyper parameters
        self._layers_size: List[int] = []
        self._n_x: int = n_x
        self._layers_activation_func: List[BaseActivation] = []
        self._cached_activations = []
        self._init_random_weight = init_random_weight
        self._prediction_proba_threshold = prediction_proba_threshold
        self._optimizer: OptimizerBase = optimizer or AdaptiveGradientDescentMomentum()
        self._cost_function: BaseLossFunction = cost_function or CrossEntropyLoss()

        # Parse layer configuration
        for layer_size, activation_func in layers_config:
            assert (issubclass(activation_func, BaseActivation))
            self._layers_size.append(layer_size)
            self._layers_activation_func.append(activation_func())

        # Model parameters
        self._layers_parameters: List[Tuple[np.ndarray, np.ndarray]] = None

        # Model cache
        self._layer_values = None

        self._initialize_network()

    def _initialize_network(self):

        # Initialize layer parameters
        self._layers_parameters = [     # (W, b)
            (np.random.randn(n, n_previous) * self._init_random_weight, np.zeros((n, 1)))
            for n, n_previous in zip(self._layers_size, [self._n_x] + self._layers_size)
        ]

    @property
    def optimizer(self) -> OptimizerBase:
        return self._optimizer

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Perform a forward propagation on the neural network
        :param X: A 2-D array (n, m) of m samples with n features.
        :return: The response value of the model (Y)
        """

        A_previous = X
        self._layer_values: List[Tuple] = [(None, X)]  # (Z, A)
        for (W, b), activation_func in zip(self._layers_parameters, self._layers_activation_func):
            Z = np.dot(W, A_previous) + b       # Calculate linear output
            A = activation_func(Z)              # Calculate activation function

            # Save to layers cache
            self._layer_values.append((Z, A))

            # Change previous A and continue
            A_previous = A

        return A_previous

    def backwards(self, Y: np.ndarray) -> np.ndarray:
        """
        Perform a backward propagation on the neural network
        :param Y: The expected outcome of the neural network
        :return: The gradients of all parameters starting from the left till the right most layer.
        """

        reversed_layer_values = list(reversed(self._layer_values))
        reversed_layer_params = list(reversed(self._layers_parameters))
        reversed_activation_functions = list(reversed(self._layers_activation_func))
        m = Y.shape[1]  # number of samples

        # Special case for last layer (first in back-propagation)
        _, A_last = reversed_layer_values[0]
        dZ_last = A_last - Y
        A_next = reversed_layer_values[1][1]
        dW_last = np.dot(dZ_last, A_next.T) / m
        db_last = np.mean(dZ_last, axis=1, keepdims=True)

        grads = [  # (dZ, dW, dB)
            (dZ_last, dW_last, db_last)
        ]
        dZ_previous = dZ_last

        for act_func, lvalues, lvalues_prev, lparams_prev in zip(
                reversed_activation_functions[1:],
                reversed_layer_values[1:],
                reversed_layer_values[2:],
                reversed_layer_params[0:]
        ):
            Z, A = lvalues
            _, A_next = lvalues_prev
            W_previous, b_previous = lparams_prev

            dZ = np.dot(W_previous.T, dZ_previous) * act_func.derivative(A)
            dW = np.dot(dZ, A_next.T) / m
            db = np.mean(dZ, axis=1, keepdims=True)
            grads.append((dZ, dW, db))

            dZ_previous = dZ

        return np.array(list(reversed(grads)))

    def loss(self, A:np.ndarray, Y:np.ndarray) -> np.ndarray:
        """
        Calculate the loss between two vectors
        :param A: The response of the model
        :param Y: The expected output
        :return: A scalar value per sample of the loss
        """
        return self._cost_function(A, Y)

    def train(self, X: np.ndarray, Y: np.ndarray, iterations: int = 100) -> np.ndarray:
        """
        Train neural network on a given dataset. Training will be performed in batch mode,
        processing the whole dataset in one vectorized command per iteration.
        :param X: A 2-D array (n, m) with of m samples with n features.
        :param Y: The expected output of the model in (n, m) format where n is the number of
        output features and m the number of samples
        :param iterations: The number of iterations to optimize parameters of the network
        :return: The final cost per iteration
        """
        from itertools import chain

        costs = []
        for iter_count in range(iterations):
            self.forward(X)

            grads = self.backwards(Y)

            cost = self.loss(self._layer_values[-1][1], Y)
            self._optimizer.update_loss(cost)

            params_and_grads = list(chain.from_iterable([
                (
                    (layer_curr_params[0], layer_grads[1]),
                    (layer_curr_params[1], layer_grads[2])
                )
                for layer_curr_params, layer_grads in zip(self._layers_parameters, grads)
            ]))

            new_params_flatten = self._optimizer.step(
                map(lambda pg: pg[0], params_and_grads),
                map(lambda pg: pg[1], params_and_grads))

            self._layers_parameters = [
                (new_params_flatten[i], new_params_flatten[i + 1])
                for i in range(0, len(new_params_flatten), 2)
            ]

            costs.append(cost)
            if iter_count % 1000 == 0:
                print(f"Iteration: {iter_count}, Cost: {self._optimizer.best_loss}")
        return np.array(costs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output class for each input feature sample
        :param X: A 2-D array (n, m) with of m samples with n features.
        :return:
        """
        A_last = self.forward(X)
        return np.array(A_last > self._prediction_proba_threshold, dtype='int')
