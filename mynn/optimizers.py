import logging as _logging
from typing import List, Optional, Iterator, Any
import numpy as np

from ._const import BIG_FLOAT, SMALL_FLOAT


logger = _logging.getLogger(__name__)


class OptimizerBase:
    """
    Base class for optimizers
    """

    def __init__(self, **params: Optional[float]):
        """
        Initialize optimizer by declaring its parameters
        :param params: A dictionary of parameters
        """
        self._last_loss = BIG_FLOAT
        self.best_loss = BIG_FLOAT
        self.best_solution = None
        self.params = params

    def __getattr__(self, item):
        if item in self.params:
            return self.params[item]
        raise AttributeError()

    def reset(self) -> None:
        """
        Reset optimizer before starting training
        :return:
        """
        self._last_loss = BIG_FLOAT
        self.best_loss = BIG_FLOAT
        self.best_solution = None

    def step(self, values: Iterator[np.ndarray],
             grads: Iterator[np.ndarray],
             epoch: int,
             mini_batch: int,
             iteration: int) -> List[np.ndarray]:
        """
        Perform an optimization iteration to estimate better values
        :param values: The values to be optimize
        :param grads: The gradients of the values
        :param epoch: The epoch that this step was processed
        :param mini_batch: The mini_batch index of that this step was processed. This is always 0 in batch mode.
        :param iteration: The actual optimization iteration
        :return: The values optimized
        """
        raise NotImplementedError()

    def update_loss(self, loss: float, solution: Any) -> None:
        """
        Update the computed loss of the last values
        :param loss:
        """
        self._last_loss = loss
        if loss < self.best_loss:
            self.best_loss = min(self.best_loss, loss)
            self.best_solution = solution

    def parameters_hash(self) -> str:
        """
        Get a hash of the parameters of the optimizer. Usefull for
        caching purposes
        """
        import hashlib
        base_str = (self.__class__.__name__ + repr(self.params))
        return hashlib.sha224(base_str.encode('utf-8')).hexdigest()


class GradientDescent(OptimizerBase):
    """
    Implementation of classic gradient descent
    """

    def __init__(self, learning_rate: float, **extra_params):
        """
        Initialize optimizer
        :param learning_rate: A rate to adapt to graadients
        """
        super().__init__(learning_rate=learning_rate, **extra_params)

    def reset(self) -> None:
        super().reset()

    def step(self, values: Iterator[np.ndarray],
             grads: Iterator[np.ndarray],
             epoch: int,
             mini_batch: int,
             iteration: int) -> List[np.ndarray]:
        return [
            v - self.learning_rate * g
            for v, g in zip(values, grads)
        ]

    def __repr__(self):
        return f"GD(rate={self.learning_rate})"


class GradientDescentMomentum(GradientDescent):
    """
    Implementation of Gradient Descent with momentum
    """

    def __init__(self, learning_rate, beta: Optional[float] = 0.9, **extra_params):
        """
        Initialize gradient descent optimizer
        :param learning_rate: The base learning rate to adapt on gradient values
        :param beta: The beta factor of the exponentially weighted averages
        """
        super().__init__(learning_rate=learning_rate, beta=beta, **extra_params)
        self._average_gradients = None

    def reset(self) -> None:
        super().reset()
        self._average_gradients = None

    def _get_updated_averages(self, grads: List[np.ndarray], iteration: int) -> List[np.ndarray]:
        """
        Get the updated gradients exponentially weighted averages
        :param grads:
        :param iteration: The iteration step, to be used for bias correction
        :return:
        """
        if self._average_gradients is None:
            return [
                np.zeros(g.shape)
                for g in grads
            ]

        return [
            (average_grads * self.beta + g * (1.0 - self.beta)) / (1 - self.beta**iteration)

            for average_grads, g in zip(self._average_gradients, grads)
        ]

    def step(self, values: Iterator[np.ndarray],
             grads: Iterator[np.ndarray],
             epoch: int,
             mini_batch: int,
             iteration: int) -> List[np.ndarray]:
        grads = list(grads)
        values = list(values)
        self._average_gradients = self._get_updated_averages(grads, iteration + 1)

        results = [
            v - self.learning_rate * grad
            for v, grad in zip(values, self._average_gradients)
        ]

        return results

    def __repr__(self):
        return f"GDMomentum(learning_rate={self.learning_rate},beta={self.beta})"


class RMSProp(GradientDescent):
    """
    Implementation of Root-Mean-Squared prop optimization algorithm
    """

    def __init__(self, learning_rate, beta2: Optional[float] = 0.9, **extra_params):
        """
        Initialize gradient descent optimizer
        :param learning_rate: The base learning rate to adapt on gradient values
        :param beta: The beta2 factor of the rmsprop
        """
        super().__init__(learning_rate=learning_rate, beta2=beta2, **extra_params)
        self._squared_average_gradients = None

    def reset(self) -> None:
        super().reset()
        self._squared_average_gradients = None

    def _get_update_squared_averages(self, grads: List[np.ndarray], iteration: int) -> List[np.ndarray]:
        """
        Get the updated exponentially weighted averages of squared gradients
        :param grads:
        :param iteration: The iteration step, to be used for bias correction
        :return:
        """
        if self._squared_average_gradients is None:
            return [
                np.zeros(g.shape)
                for g in grads
            ]

        return [
            (average_grads * self.beta2 + (g**2) * (1.0 - self.beta2)) / (1 - self.beta2**iteration)
            for average_grads, g in zip(self._squared_average_gradients, grads)
        ]

    def step(self, values: Iterator[np.ndarray],
             grads: Iterator[np.ndarray],
             epoch: int,
             mini_batch: int,
             iteration: int) -> List[np.ndarray]:
        grads = list(grads)
        values = list(values)
        self._squared_average_gradients = self._get_update_squared_averages(grads, iteration + 1)

        results = [
            v - self.learning_rate * grad/np.sqrt(average_grad + SMALL_FLOAT)
            for v, average_grad, grad in zip(values, self._squared_average_gradients, grads)
        ]

        return results

    def __repr__(self):
        return f"RMSProp(learning_rate={self.learning_rate},beta2={self.beta2})"


class Adam(RMSProp, GradientDescentMomentum):
    """
    Implementation of Adam algorithm (RMSProp + Momentum)
    """

    def __init__(self, learning_rate, beta: Optional[float] = 0.9, beta2: Optional[float] = 0.99, **extra_params):
        """
        Initialize optimizer
        :param learning_rate: The base learning rate to adapt on gradient values
        :param beta: The beta factor of the exponentially weighted averages
        :param beta2: The beta factor of the root mean squared
        """
        super().__init__(learning_rate=learning_rate, beta=beta, beta2=beta2, **extra_params)

    def step(self, values: Iterator[np.ndarray],
             grads: Iterator[np.ndarray],
             epoch: int,
             mini_batch: int,
             iteration: int) -> List[np.ndarray]:
        grads = list(grads)
        values = list(values)
        self._average_gradients = self._get_updated_averages(grads, iteration + 1)
        self._squared_average_gradients = self._get_update_squared_averages(grads, iteration + 1)

        results = [
            v - self.learning_rate * grad / (np.sqrt(sgrad)  + SMALL_FLOAT)
            for v, grad, sgrad in zip(values, self._average_gradients, self._squared_average_gradients)
        ]

        return results

    def __repr__(self):
        return f"Adam(learning_rate={self.learning_rate},beta={self.beta},beta2={self.beta2})"


class AdaptiveGradientDescentMomentum(GradientDescentMomentum):
    """
    Implementation of Gradient Descent with adaptive learning rate and momentum
    """

    def __init__(self, min_learning_rate: float = 1.2, max_learning_rate: float = 5,
                 beta: Optional[float] = 0.9):
        """
        Initialize optimizer
        :param min_learning_rate: The maximum learning rate to be used. This is also the starting learning rate
        :param max_learning_rate: The minimum learning rate to be used.
        :param beta: The beta factor of the exponentially weighted averages
        """
        super().__init__(learning_rate=max_learning_rate,
                         min_learning_rate=min_learning_rate,
                         max_learning_rate=max_learning_rate,
                         beta=beta)
        self._old_learning_rates = None
        self._old_grads = None

    def reset(self) -> None:
        super().reset()
        self._old_learning_rates = None
        self._old_grads = None

    def step(self, values: Iterator[np.ndarray],
             grads: Iterator[np.ndarray],
             epoch: int,
             mini_batch: int,
             iteration: int) -> List[np.ndarray]:

        grads = list(grads)
        values = list(values)
        new_averages = self._get_updated_averages(grads)
        if self._average_gradients is None:
            self._average_gradients = new_averages
            self._old_learning_rates = [np.ones(lr.shape) * self.max_learning_rate for lr in values]

        learning_rates = []
        results = []
        for v, g, old_g, learning_rate in zip(values, new_averages, self._average_gradients, self._old_learning_rates):

            # If the optimizing is oscillating then reduce learning rate, else slightly increase.
            has_not_changed_side = np.equal(g / np.abs(g), old_g / np.abs(old_g))
            learning_rate = np.where(
                has_not_changed_side,
                np.minimum(self.max_learning_rate, learning_rate * 1.07),
                np.maximum(self.min_learning_rate, learning_rate * 0.5)
            )

            results.append(v - learning_rate * g)
            learning_rates.append(learning_rate)

        self._average_gradients = new_averages
        self._old_learning_rates = learning_rates

        return results

    def __repr__(self):
        return f"Adaptive(rate={self.min_learning_rate} - {self.max_learning_rate}],beta={self.beta})"
