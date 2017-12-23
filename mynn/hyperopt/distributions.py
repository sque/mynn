import numpy as np
from typing import List, Union
from abc import ABCMeta, abstractmethod


class Distribution(metaclass=ABCMeta):
    """
    Base class for declaring distributions
    """

    @abstractmethod
    def _sampler_impl(self, n: int) -> np.ndarray:
        """
        Randomly take N samples from the distribution
        :param n: The number of samples to return.
        """
        pass

    def sample(self, n: int = None) -> Union[np.ndarray, float, int]:
        """
        Randomly take N samples from the distribution
        :param n: The number of samples to return. If None it will return
        just one number and not a vector.
        """
        if n is None:
            return self._sampler_impl(1)[0]
        else:
            return self._sampler_impl(n)


class UniformDistribution(Distribution):
    """
    Uniform distribution in a linear space
    """

    def __init__(self, min_, max_):
        """
        Initialize distribution
        :param min_: The lower boundary of the distribution
        :param max_: The upper boundary of the distribution
        """
        self.min_ = min_
        self.max_ = max_
        assert (min_ < max_)

    @property
    def delta(self) -> float:
        """
        The delta of the distribution
        """
        return self.max_ - self.min_

    def _sampler_impl(self, n: int) -> np.ndarray:
        return np.random.rand(n) * self.delta + self.min_


uniform = UniformDistribution


class LogUniformDistribution(Distribution):
    """
    Uniform distribution in the log scale of a linear variable
    """

    def __init__(self, min_, max_):
        """
        :param min_: The minimum value of the distribution
        :param max_: The maximum value of the distribution
        """
        self.min_ = min_
        self.max_ = max_

        assert (min_ >= 0)
        assert (max_ >= 0)
        assert (min_ < max_)

        self.min_log = np.log10(self.min_)
        self.max_log = np.log10(self.max_)

    @property
    def delta(self) -> float:
        """Get the delta of the linear space"""
        return self.max_ - self.min_

    @property
    def delta_log(self) -> float:
        """Get the delta of the log space"""
        return self.max_log - self.min_log

    def _sampler_impl(self, samples: int):
        log_sample = (np.random.rand(samples) * self.delta_log + self.min_log)
        return 10 ** log_sample


log_uniform = LogUniformDistribution


class UniformChoiceDistribution(Distribution):
    """
    Uniform distribution for preselected choices
    """

    def __init__(self, choices: List) -> None:
        """
        Initialize distribution
        :param choices: A pool of choice to uniformly sample
        """
        self.choices_ = choices

    def _sampler_impl(self, samples: int) -> np.ndarray:
        return np.random.choice(self.choices_, samples)


uniform_choice = UniformChoiceDistribution


class DiscreteDistribution(Distribution):
    """
    Meta-distribution to convert any continuous space to discrete
    """

    def __init__(self, source: Distribution) -> None:
        """
        Initialize discrete modifier
        :param source: A source distribution of continuous space
        """
        self.source = source

    def _sampler_impl(self, samples: int = 1):
        return np.array(
            np.rint(self.source.sample(samples)),
            dtype='int')


discrete = DiscreteDistribution
