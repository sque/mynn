from .distributions import Distribution, uniform, discrete, log_uniform, uniform_choice
from typing import List, Dict
from datetime import timedelta


class Configuration(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._test_score = None

    @property
    def test_score(self):
        return self._test_score

    def update_score(self, score):
        self._test_score = score


class ExecutionIteration:

    def __init__(self, configuration: Dict):
        self.config = configuration
        self._test_score = None
        self._running_time = None
        self._score = None

    @property
    def test_score(self):
        return self._test_score

    def done(self, running_time: timedelta, score: float, meta: Dict):
        self._test_score = score
        self._running_time = running_time
        self._test_score = score


class RandomOptimizer:

    def __init__(self, parameters: Dict[str, distributions.Distribution] = None):
        self._parameters: Dict[str, distributions.Distribution] = {}
        if parameters is not None:
            self._parameters = parameters

    def next_execution(self) -> ExecutionIteration:
        """Get the next proposed execution
        """
        config = {
            name: param.sample()
            for name, param in self._parameters.items()
        }

        return ExecutionIteration(config)
