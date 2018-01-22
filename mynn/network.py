import logging as _logging
from typing import List, Optional, Tuple, ContextManager, Callable, Union, Dict, Generator
from contextlib import contextmanager

import numpy as np
from prettytable import PrettyTable

from .optimizers import OptimizerBase, Adam
from .initializers import WeightInitializerBase, XavierWeightInitializer
from .endecoders import LabelEncoderDecoder
from .loss import BaseLossFunction, BinaryCrossEntropyLoss
from . import _utils
from .regularization import RegularizationBase
from .layers import  Layer

logger = _logging.getLogger(__name__)




class NNModel:
    """
    Neural Network model
    """

    def __init__(self,
                 inputs,
                 outputs,
                 loss_function: BaseLossFunction,
                 output_encoder_decoder: Optional[LabelEncoderDecoder] = None,
                 optimizer: Optional[OptimizerBase] = None,
                 verbose_logging: bool = False,
                 weights_initializer: Optional[WeightInitializerBase] = False,
                 regularization: Optional[RegularizationBase] = None):
        """
        Initialize an untrained FNN.
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
        :param weights_initializer: The weight initializer algorithm. If None the default Xavier will be
        :param regularization: A regularization method to be applied on the model.
        used.
        """

        self._input_layer = inputs
        self._output_layer: Layer = outputs
        self._layers_index: Dict[str, Layer] = {}

        # Hyper parameters
        self._output_encoder_decoder = output_encoder_decoder
        self._optimizer: OptimizerBase = optimizer or Adam(learning_rate=0.001)
        self._weights_initializer: WeightInitializerBase = weights_initializer or XavierWeightInitializer()
        self._loss_function: BaseLossFunction = loss_function or BinaryCrossEntropyLoss()
        self._regularization = regularization
        self._enabled_regularization: bool = False

        self._verbose_logging = verbose_logging

        self._initialize_network()
        logger.debug(f"Initialized NN network model")
        logger.debug(f"  Layers: max-depth={max(map(lambda l: l.depth, self._layers_index.values()))} total=#{len(self._layers_index)}")
        logger.debug(f"  Optimizer: {self._optimizer}")
        logger.debug(f"  Weight Initializer: {self._weights_initializer}")
        logger.debug(f"  Regularization: {self._regularization}")
        if self._output_encoder_decoder:
            logger.debug(f"  Encoder/Decoder: {self._output_encoder_decoder}")

    def _initialize_network(self):
        """
        Initialize the network components so that they are ready for training
        """

        # Walk thourgh all layers and initialize them
        for layer, depth in self._layers_forward_order(return_depth=True):

            # Set appropriate name for all layers
            if layer.name is None:
                layer.name = f"{layer.layer_type()}_{depth}"

            # Store layer in index
            if layer.name in self._layers_index:
                raise KeyError(f"There are two layers with the same name: \"{layer.name}\"")
            self._layers_index[layer.name] = layer

            # Update its depth
            layer.depth = depth

            # Initialize parameters of layer
            self._weights_initializer(layer)


    def _layers_forward_order(self, return_depth:bool = False) -> Union[List[Layer], List[Tuple[Layer,int]]]:
        """
        Iterator to walk from input to output layer
        :param return_depth: If true it will return the layer and its depth for each layer.
        """
        layers = list(reversed(list(self._layer_backward_order(return_reverse_depth=return_depth))))

        if not return_depth:
            return layers

        # Find maximum depth
        max_depth = abs(min(map(lambda tup: tup[1], layers)))

        return list(map(lambda tup: (tup[0], max_depth + tup[1]), layers))

    def _layer_backward_order(self, starting_layer: Optional[Layer] = None,
                              return_reverse_depth:bool = False,
                              current_reverse_depth:Optional[int] = None) -> Generator[Layer, None, None]:
        """
        Iterator to walk from all layers starting from output till input layer.
        For multiple paths it will follow a breadth-first approach. Optionally you can also
        ask for the depth of each layer, in reverse order where 0 is the last and -X is the first one.
        :param starting_layer: The layer to start traversing backwards. If None it will use models output layer
        :param return_reverse_depth: If true it will return along with the layer the depth
        :param current_reverse_depth: The order of the current layer
        :return:
        """
        if starting_layer is None:
            starting_layer = self._output_layer

        if current_reverse_depth is None:
            current_reverse_depth = 0

        if return_reverse_depth:
            yield starting_layer, current_reverse_depth
        else:
            yield starting_layer

        if starting_layer.input_layers:
            for layer in starting_layer.input_layers:
                if layer is None:
                    continue
                yield from self._layer_backward_order(starting_layer=layer,
                                                      return_reverse_depth=return_reverse_depth,
                                                      current_reverse_depth=current_reverse_depth - 1)

    def layers_summary(self):
        """
        A string that describe all layers of the model
        """
        t = PrettyTable()
        t.field_names = ["Type", "Name", "Output shape", "Inputs"]
        t.align["Type"] = 'l'

        for layer in self._layers_forward_order():
            input_names = ",".join([input_l.name for input_l in layer.input_layers])

            t.add_row([
                str(layer),
                layer.name,
                layer.output_shape,
                input_names,

            ])

        return t

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

        layer_outputs = {
            self._input_layer.name: X
        }

        # Walk through all layers and propagate values
        for layer in self._layers_forward_order():
            if layer.name in layer_outputs:
                continue

            # Resolve dependencies
            in_layer_names = [in_layer.name for in_layer in layer.input_layers]

            # Prepare input
            in_values = [layer_outputs[layer_name] for layer_name in in_layer_names]

            # forward pass through the layer
            output = layer.forward(*in_values)

            # store results
            layer_outputs[layer.name] = output

        # When all layer are executed return the output layer result
        return layer_outputs[self._output_layer.name]

    def backwards(self, Y: np.ndarray) -> Dict[str, Tuple]:
        """
        Perform a backward propagation on the neural network
        :param Y: The expected outcome of the neural network
        :return: The gradients of parameters per layer
        """

        if self._verbose_logging:
            logger.debug(f"Performing backwards propagation for Y:{Y.shape}")

        m = Y.shape[1]      # number of samples

        # Registry of all back propagations
        d_outputs = {
            self._output_layer.name: self._loss_function.derivative(self._output_layer.cached_output, Y)  # This cached output probably should be stored in loss layer
        }

        d_layer_parameters = {}

        for layer in self._layer_backward_order():

            # Find layers incoming output gradient from backpropagation
            dOut = d_outputs[layer.name]

            # Back-propagate from the current layer
            d_inputs, d_parameters = layer.backward(dOut)

            # Store input derivatives for the input layer
            for in_layer, d_input in zip(layer.input_layers, d_inputs):
                d_outputs[in_layer.name] = d_input

            # Store parameter gradients for training
            d_layer_parameters[layer.name] = d_parameters

        return d_layer_parameters

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
              post_iteration_callbacks: List[Callable[['NNModel', _utils.TrainingContext], bool]] = None,
              train_ctx: _utils.TrainingContext = None,
              log_every_nth: int = 25) -> _utils.TrainingContext:
        """
        Train neural network on a given dataset. Training will be performed in batch mode,
        processing the whole dataset in one vectorized command per iteration.
        :param X: A 2-D array (n, m) with of m samples with n features.
        :param Y: The expected output of the model in (n, m) format where n is the number of
        output features and m the number of samples
        :param epochs: The number of epochs to optimize parameters of the network
        :param mini_batch_size: The size of mini-batches to perform optimization. If None, it will perform batch
        optimization
        (neural_network, epoch, mini_batch, iteration). The output of the function will be attached to the returned
        costs.
        :param post_iteration_callbacks: A list of callbacks that will be triggered after each
        iteration. Usually to calculate extra costs or early stop.
        :param log_every_nth: Every nth iteration will log the current cost
        :return: The final cost per iteration
        """
        timer = _utils.RelativeTime()

        logger.debug(f"Training on X: {X.shape}, Y: {Y.shape} for {epochs} epochs.")

        if X.shape[-1] != Y.shape[-1]:
            raise ValueError("X and Y dataset do not contain the same number of samples.")

        if mini_batch_size is not None:
            logger.debug(f"Enabling mini-batch optimization with size={mini_batch_size}")

        if self._output_encoder_decoder:
            # Encode Y variable
            logger.debug(f"Encoding Y variable")
            Y = self._output_encoder_decoder.encode(Y)

        self._optimizer.reset()
        ctx: _utils.TrainingContext = train_ctx or _utils.TrainingContext()

        for epoch in ctx.iter_epochs(epochs):
            if self._verbose_logging:
                logger.debug(f"Starting train iteration: {epoch} ")

            for X_batch, Y_batch in \
                    ctx.iter_mini_batch(_utils.random_mini_batches(X=X, Y=Y, mini_batch_size=mini_batch_size)):

                with self.training_mode():
                    # Forward and then backwards propagation to generate the gradients
                    self.forward(X_batch)
                    d_parameters = self.backwards(Y_batch)

                    # Parameters and grads are stored in named tuples per layer. Optimizer excepts a list
                    # values and their gradients in order to optimize. In the next two blocks we unpack
                    # parameters in a flat format to optimize them and then store them to layers

                    values = []     # List of parameters to be optimize
                    grads = []      # List of equivalent grads for the above parameters
                    origin = []     # Tuples (layer, param_name) of parameters origin
                    for layer_name, layer_grads in d_parameters.items():
                        layer = self._layers_index[layer_name]
                        for (param_name, param_value), grad in zip(layer.parameters.items(), layer_grads):
                            grads.append(grad)
                            values.append(param_value)
                            origin.append([layer.name, param_name])

                    # Calculate loss and give a chance to the optimizer to do something smarter
                    cost = self.loss(self._output_layer.cached_output, Y_batch)

                    # Parameters and grads are stored in named tuples per layer. Optimizer does not
                    # understand this structure but expects an iterable for values and for grads. In
                    # the next two blocks we unpack parameters in a flat format to optimize them and
                    # then repack to store them.

                    # Unpack parameters and grads and trigger optimizer step
                    new_params_flatten = self._optimizer.step(
                        values=values,
                        grads=grads,
                        epoch=ctx.current_epoch,
                        mini_batch=ctx.current_mini_batch_index,
                        iteration=ctx.current_iteration_index
                    )

                    # Repack and update model parameters
                    for (layer_name, param_name), param_value in zip(origin, new_params_flatten):
                        self._layers_index[layer_name].parameters = {param_name: param_value}

                # Finish iteration
                ctx.iteration_done(cost)

                # Call callbacks
                if post_iteration_callbacks:
                    should_stop = []
                    for cb in post_iteration_callbacks:
                        cb_response = cb(self, ctx)
                        should_stop.append(cb_response)

                    if any(should_stop):
                        logger.debug("Requested to early stop from post iteration callback")
                        return ctx

                if ctx.current_iteration_index % log_every_nth == 0:
                    logger.debug(
                        f"[{timer.passed_timedelta!s}|ep:{ctx.current_epoch}|mb:{ctx.current_mini_batch_index}] "
                        f"Current cost {cost}.")
                    # Log progress and report estimated time
            logger.debug(f"[{timer.passed_timedelta!s}|ep:{ctx.current_epoch}|mb:{ctx.current_mini_batch_index}] "
                         f"Current cost {cost}. Finishing in {ctx.estimated_remaining_time}.")

        logger.debug(f"Finished training in {ctx.passed_timedelta!s} after {ctx.current_epoch+ 1}"
                     f" epochs with final cost: {cost}")

        return ctx

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

