import logging as _logging
from typing import List, Optional, Tuple, ContextManager, Callable
from contextlib import contextmanager

import numpy as np

from .optimizers import OptimizerBase, Adam
from .activation import BaseActivation, SoftmaxActivation, SigmoidActivation
from .initializers import WeightInitializerBase, VarianceScalingWeightInitializer
from .endecoders import LabelEncoderDecoder
from .loss import BaseLossFunction, CrossEntropyLoss, BinaryCrossEntropyLoss
from . import _utils
from .value_types import LayerParameters, LayerValues, LayerGrads
from .regularization import RegularizationBase


logger = _logging.getLogger(__name__)


class FNN:
    """
    Simple implementation of L Feedforward Neural Network
    """

    def __init__(self, layers_config: List[Tuple[int, BaseActivation]],
                 n_x: int,
                 output_encoder_decoder: Optional[LabelEncoderDecoder] = None,
                 optimizer: Optional[OptimizerBase] = None,
                 loss_function: Optional[BaseLossFunction] = None,
                 verbose_logging: bool = False,
                 initializer: Optional[WeightInitializerBase] = False,
                 regularization: Optional[RegularizationBase] = None):
        """
        Initiate an untrained FNN. Notation and index of layers is according to Andrew's NG
        lesson for Neural Networks. So layer 1 is the input layer and layer 0 is a pseudo-layer
        for input data.
        :param layers_config: A list of tuples describing each layer starting from the first hidden till the output
        layer. The tuple must consist of the number of nodes and the class of the activation function to use.
        :param n_x: Number of input features
        :param output_encoder_decoder: An encoder/decoder that will be used to encode y_variable before training
        and decode on predict() function.
        :param optimizer: The optimizer object to use for optimization. If not defined it will use the Adaptive GD
        with default parameters.
        :param loss_function: The loss function to use for training the neural network. If not set the default
        CrossEntropyLoss will be used.
        :param verbose_logging: A flag, where if True it will enable logging extract debug information under DEBUG
        level. This is disabled by default for performance reasons.
        :param initializer: The weight initializer algorithm. If None the default VarianceScaling of scale=2 will be
        :param regularization: A regularization method to be applied on the model.
        used.
        """

        # Hyper parameters
        self._layers_size: List[int] = []
        self._n_x: int = n_x
        self._layers_activation_func: List[BaseActivation] = []
        self._cached_activations = []
        self._output_encoder_decoder = output_encoder_decoder
        self._optimizer: OptimizerBase = optimizer or Adam(learning_rate=0.001)
        self._initializer: WeightInitializerBase = initializer or VarianceScalingWeightInitializer(scale=2)
        self._loss_function: BaseLossFunction = loss_function or BinaryCrossEntropyLoss()
        self._regularization = regularization
        self._enabled_regularization: bool = False

        # Parse layer configuration
        self._layers_size.append(n_x)
        self._layers_activation_func.append(None)
        self._layers_parameters = [LayerParameters(W=1, b=2)]
        for layer_size, activation_func in layers_config:
            assert (issubclass(activation_func, BaseActivation))
            self._layers_size.append(layer_size)
            self._layers_activation_func.append(activation_func())

        # Model parameters
        self._layers_parameters: List[LayerParameters] = []

        # Model cache
        self._layer_values: List[LayerValues] = None

        self._initialize_network()

        self._verbose_logging = verbose_logging
        logger.debug(f"Initialized FNN network of #{len(self._layers_size) - 1} layers")
        logger.debug(f"  Layers sizes: {self._layers_size[1:]}")
        logger.debug(f"  Activation functions: {self._layers_activation_func[1:]}")
        logger.debug(f"  Optimizer: {self._optimizer}")
        logger.debug(f"  Weight Initializer: {self._initializer}")
        logger.debug(f"  Regularization: {self._regularization}")
        if self._output_encoder_decoder:
            logger.debug(f"  Encoder/Decoder: {self._output_encoder_decoder}")

    def _initialize_network(self):

        # Check that we are using Sigmoid-CrossEntropy and Softmax-CrossEntropy
        # Ugly hack to override the need to analytically calculate the dA using the costly
        # cubic derivative of Softmax.
        # Can be fixed by first calculating the operations to be performed at each layer
        # on forward and backward propagation
        valid_schemas = [
            isinstance(self._loss_function, BinaryCrossEntropyLoss)
            and isinstance(self._layers_activation_func[-1], SigmoidActivation),
            isinstance(self._loss_function, CrossEntropyLoss)
            and isinstance(self._layers_activation_func[-1], SoftmaxActivation),
        ]
        if not any(valid_schemas):
            raise TypeError("You need to use a compatible output layer with loss function.")

        # Initialize layer parameters
        self._layers_parameters = [LayerParameters(None, None)]
        for l, (n, n_left) in enumerate(zip(self._layers_size[1:], self._layers_size), start=1):
            W, b = self._initializer.get_initial_weight(l=l, n=n, n_left=n_left)
            self._layers_parameters.append(LayerParameters(W=W, b=b))

    @property
    def optimizer(self) -> OptimizerBase:
        """
        Get access to optimizer object
        """
        return self._optimizer

    @property
    def output_encoder_decoder(self) -> Optional[LabelEncoderDecoder]:
        """
        Get the output encoder decoder.
        :return:
        """
        return self._output_encoder_decoder

    @contextmanager
    def training_mode(self) -> ContextManager:
        """
        A context manager to instruct network that it enter training mode.

        While in training mode, the network enables regularization
        """
        if self._verbose_logging:
            logger.debug(f"Entering training mode, enabling regularization")
        self._enabled_regularization = True
        yield
        self._enabled_regularization = False
        if self._verbose_logging:
            logger.debug(f"Leaving training mode, disabling regularization")

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Perform a forward propagation on the neural network
        :param X: A 2-D array (n, m) of m samples with n features.
        :return: The response value of the model (Y)
        """

        if self._verbose_logging:
            logger.debug(f"Performing forward propagation for X:{X.shape}")

        A_previous = X
        self._layer_values: List[Tuple] = [LayerValues(Z=None, A=X, extras={})]  # (Z, A)
        for l_index, l_params, l_activation_func in zip(
                range(1, len(self._layers_parameters) + 1),
                self._layers_parameters[1:],
                self._layers_activation_func[1:]):

            Z = np.dot(l_params.W, A_previous) + l_params.b       # Calculate linear output
            A = l_activation_func(Z)                              # Calculate activation function

            layer_values = LayerValues(Z=Z, A=A, extras={})

            # Regularization hook
            if self._regularization and self._enabled_regularization:
                layer_values = self._regularization.on_post_forward_propagation(
                    layer_values=layer_values,
                    layer_index=l_index,
                    layer_params=l_params,
                )

            # Save to layers cache
            self._layer_values.append(layer_values)

            # Change previous A and continue
            A_previous = A

        return A_previous

    def backwards(self, Y: np.ndarray) -> List[LayerGrads]:
        """
        Perform a backward propagation on the neural network
        :param Y: The expected outcome of the neural network
        :return: The gradients of all parameters starting from the left till the right most layer.
        """

        if self._verbose_logging:
            logger.debug(f"Performing backwards propagation for Y:{Y.shape}")

        m = Y.shape[1]      # number of samples
        dA_l_right = None

        grads = []
        for l_index, l_activation_func, l_params, l_values, l_left_values in zip(
                range(len(self._layer_values) - 1, 0, -1),
                reversed(self._layers_activation_func),
                reversed(self._layers_parameters),
                reversed(self._layer_values),
                reversed(self._layer_values[:-1]),
        ):
            if dA_l_right is None:
                # First iteration, take dA from loss function
                dZ = l_values.A - Y
            else:
                dZ = dA_l_right * l_activation_func.derivative(l_values.Z)

            # Regularization hook
            if self._regularization and self._enabled_regularization:
                dZ = self._regularization.on_pre_backward_propagation(
                    dZ=dZ,
                    layer_index=l_index,
                    samples=m,
                    layer_values=l_values,
                    layer_params=l_params)

            # Calculate dZ, dW, db for this layer and store them
            # dZ = dA_l_right * l_activation_func.derivative(l_values.Z)
            dW = np.dot(dZ, l_left_values.A.T) / m
            db = np.mean(dZ, axis=1, keepdims=True)

            layer_grads = LayerGrads(dW=dW, db=db)

            # Regularization hook
            if self._regularization and self._enabled_regularization:
                layer_grads = self._regularization.on_post_backward_propagation(
                    grads=layer_grads,
                    layer_index=l_index,
                    samples=m,
                    layer_values=l_values,
                    layer_params=l_params)

            grads.append(layer_grads)

            # Calculate the current dA so that it will be used in next iteration
            dA_l_right = np.dot(l_params.W.T, dZ)

        return list(reversed(grads))

    def loss(self, A: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Calculate the loss between two vectors
        :param A: The response of the model
        :param Y: The expected output
        :return: A scalar value per sample of the loss
        """
        return self._loss_function(A, Y)

    def train(self,
              X: np.ndarray,
              Y: np.ndarray,
              epochs: int = 100,
              mini_batch_size: Optional[int] = None,
              iteration_callback: Callable[['FNN', 'int', 'int', 'int'], float] = None,
              log_every_nth:int=100) -> np.ndarray:
        """
        Train neural network on a given dataset. Training will be performed in batch mode,
        processing the whole dataset in one vectorized command per iteration.
        :param X: A 2-D array (n, m) with of m samples with n features.
        :param Y: The expected output of the model in (n, m) format where n is the number of
        output features and m the number of samples
        :param epochs: The number of epochs to optimize parameters of the network
        :param mini_batch_size: The size of mini-batches to perform optimization. If None, it will perform batch
        optimization
        :param iteration_callback: A function to be called per iteration with the following arguments:
        (neural_network, epoch, mini_batch, iteration). The output of the function will be attached to the returned
        costs.
        :param log_every_nth: Every nth iteration will log the current cost
        :return: The final cost per iteration
        """

        logger.debug(f"Training on X: {X.shape}, Y: {Y.shape} for {epochs} epochs.")

        if X.shape[1] != Y.shape[1]:
            raise ValueError("X and Y dataset do not contain the same number of samples.")

        total_samples = X.shape[1]
        if mini_batch_size is None:
            # Disabling batching is like running in one batch
            mini_batch_size = total_samples
        else:
            logger.debug(f"Enabling mini-batch optimization with size={mini_batch_size}")
            shuffled_indices = np.random.permutation(total_samples)
            logger.debug(f"Shuffling dataset for improved optimization performance")
            X = X[:, shuffled_indices]
            Y = Y[:, shuffled_indices]

        if self._output_encoder_decoder:
            # Encode Y variable
            logger.debug(f"Encoding Y variable")
            Y = self._output_encoder_decoder.encode(Y)

        self._optimizer.reset()
        costs = []
        for epoch in range(epochs):
            if self._verbose_logging:
                logger.debug(f"Starting train iteration: {epoch} ")

            for batch_index, batch_start in enumerate(range(0, total_samples, mini_batch_size)):
                iteration = epoch * mini_batch_size + batch_index
                batch_end = batch_start + min(batch_start + mini_batch_size, total_samples)
                X_batch = X[:, batch_start:batch_end]
                Y_batch = Y[:, batch_start:batch_end]

                with self.training_mode():
                    # Forward and then backwards propagation to generate the gradients
                    self.forward(X_batch)
                    grads = self.backwards(Y_batch)

                    # Calculate loss and give a chance to the optimizer to do something smarter
                    cost = self.loss(self._layer_values[-1].A, Y_batch)
                    # self._optimizer.update_loss(cost, self._layers_parameters)

                    # Parameters and grads are stored in named tuples per layer. Optimizer does not
                    # understand this structure but expects an iterable for values and for grads. In
                    # the next two blocks we unpack parameters in a flat format to optimize them and
                    # then repack to store them.

                    # Unpack parameters and grads and trigger optimizer step
                    new_params_flatten = self._optimizer.step(
                        values=list(_utils.nested_chain_iterable(self._layers_parameters[1:], 1)),
                        grads=list(_utils.nested_chain_iterable(grads, 1)),
                        epoch=epoch,
                        mini_batch=batch_index,
                        iteration=iteration
                    )

                    # Repack and update model parameters
                    for l, parameters in enumerate(_utils.grouped(new_params_flatten, 2), 1):
                        self._layers_parameters[l] = LayerParameters(
                            W=parameters[0], b=parameters[1]
                        )

                if iteration_callback:
                    extra_cost = iteration_callback(self, epoch, batch_index, iteration)
                    costs.append((cost, extra_cost))
                else:
                    costs.append(cost)
                if self._verbose_logging or (iteration % log_every_nth == 0):
                    logger.debug(f"Epoch: {epoch}, Batch: {batch_index}, Cost: {cost}")

        logger.debug(f"Finished training after {epoch + 1} epochs with final cost: {cost}")

        return np.array(costs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output class for each input feature sample
        :param X: A 2-D array (n, m) with of m samples with n features.
        :return:
        """
        A_last = self.forward(X)
        if self.output_encoder_decoder is None:
            return A_last

        # Decode neural network output and return results
        return self.output_encoder_decoder.decode(A_last)
