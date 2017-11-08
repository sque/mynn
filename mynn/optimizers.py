from typing import List, Optional, Iterator
import numpy as np

from ._const import BIG_FLOAT


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
        self.params = params

    def __getattr__(self, item):
        if item in self.params:
            return self.params[item]
        raise AttributeError()

    def step(self, values: Iterator[np.ndarray], grads: Iterator[np.ndarray]) -> List[np.ndarray]:
        """
        Perform an optimization iteration to estimate better values
        :param values: The values to be optimize
        :param grads: The gradients of the values
        :return: The values optimized
        """
        raise NotImplementedError()

    def update_loss(self, loss: float):
        """
        Update the computed loss of the last values
        :param loss:
        """
        self._last_loss = loss
        self.best_loss = min(self.best_loss, loss)

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

    def step(self, values: Iterator[np.ndarray], grads: Iterator[np.ndarray]) -> List[np.ndarray]:
        return [
            v - self.learning_rate * g
            for v, g in zip(values, grads)
        ]

    def __str__(self):
        return f"GD[rate={self.learning_rate}]"


class GradientDescentMomentum(GradientDescent):
    """
    Implementation of Gradient Descent with momentum
    """

    def __init__(self, learning_rate, old_grad_percent: Optional[float] = 0.2):
        """
        Initialize gradient descent optimizer
        :param learning_rate: The base learning rate to adapt on gradient values
        :param old_grad_percent: The percentage to use the previous values gradients
        """
        super().__init__(learning_rate=learning_rate, old_grad_percent=old_grad_percent)
        self._old_grads = None

    def step(self, values: Iterator[np.ndarray], grads: Iterator[np.ndarray]) -> List[np.ndarray]:
        grads = list(grads)
        values = list(values)
        if self._old_grads is None:
            self._old_grads = grads
            return super().step(values, grads)

        results = [
            v - self.learning_rate * (g * (1.0 - self.old_grad_percent) + old_g * self.old_grad_percent)
            for v, g, old_g in zip(values, grads, self._old_grads)
        ]

        self._old_grads = grads

        return results

    def __str__(self):
        return f"GDMomentum[rate={self.learning_rate},old_%={self.old_grad_percent}]"


class AdaptiveGradientDescentMomentum(GradientDescent):
    """
    Implementation of Gradient Descent with adaptive learning rate and momentum
    """

    def __init__(self, min_learning_rate: float = 1.2, max_learning_rate: float = 5,
                 old_grad_percent: Optional[float] = 0.3):
        """
        Initialize optimizer
        :param min_learning_rate: The maximum learning rate to be used. This is also the starting learning rate
        :param max_learning_rate: The minimum learning rate to be used.
        :param old_grad_percent: The percentage to use the previous values gradients
        """
        super().__init__(learning_rate=max_learning_rate,
                         min_learning_rate=min_learning_rate,
                         max_learning_rate=max_learning_rate,
                         old_grad_percent=old_grad_percent)
        self._old_learning_rates = None
        self._old_grads = None

    def step(self, values: Iterator[np.ndarray], grads: Iterator[np.ndarray]) -> List[np.ndarray]:

        grads = list(grads)
        values = list(values)
        if self._old_learning_rates is None:
            print("old learning rates")
            self._old_learning_rates = [np.ones(lr.shape) * self.max_learning_rate for lr in values]
            self._old_grads = grads
            return super().step(values, grads)

        learning_rates = []
        results = []
        for v, g, old_g, learning_rate in zip(values, grads, self._old_grads, self._old_learning_rates):

            # If the optimizing is oscillating then reduce learning rate, else slightly increase.
            has_not_changed_side = np.equal(g / np.abs(g), old_g / np.abs(old_g))
            learning_rate = np.where(
                has_not_changed_side,
                np.minimum(self.max_learning_rate, learning_rate + (learning_rate / 16)),
                np.maximum(self.min_learning_rate, learning_rate * 0.7)
            )

            if self.old_grad_percent is None:
                results.append(v - learning_rate * g)
            else:
                results.append(v - learning_rate * (g * (1.0 - self.old_grad_percent) + old_g * self.old_grad_percent))
            learning_rates.append(learning_rate)

        self._old_grads = grads
        self._old_learning_rates = learning_rates

        return results

    def __str__(self):
        return f"Adaptive[rate={self.min_learning_rate} - {self.max_learning_rate}],old_%={self.old_grad_percent}]"