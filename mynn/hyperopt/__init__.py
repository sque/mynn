import numpy as np
from .distributions import Distribution, uniform, discrete, log_uniform, uniform_choice
from typing import List, Dict
from datetime import timedelta
from enum import Enum, unique
from .._utils import RelativeTime
import logging
import pickle


@unique
class ExecutionState(Enum):
    NOT_STARTED = 1
    FINISHED = 2
    ABORTED = 3


class ExecutionIteration:

    def __init__(self, configuration: Dict):
        self._started_at = RelativeTime()
        self.config = configuration
        self._elapsed_time = None
        self._state: ExecutionState = ExecutionState.NOT_STARTED
        self._abort_reason = None
        self._finished_meta = None
        self._score = None
        logging.debug("New optimization execution has started.")

    @property
    def state(self) -> ExecutionState:
        return self._state

    @property
    def score(self):
        return self._score

    def evaluate(self, objective_function):
        self._started_at.reset()
        try:
            score, meta = objective_function(self.config)
            self.done(score, meta)
        except Exception as e:
            self.abort(str(e))
        return self

    def done(self, score: float, meta: Dict = None):
        self._score = score
        self._state = ExecutionState.FINISHED
        self._finished_meta = meta
        self._elapsed_time = self._started_at.passed_timedelta
        logging.info(f"Optimization execution finished in {self._elapsed_time} "
                     f"with {self.score} score")

    def abort(self, reason):
        self._state = ExecutionState.ABORTED
        self._abort_reason = reason
        logging.info(f"Optimization execution aborted because \"{reason}\"")

    def __repr__(self):
        return f"<Exec {self._state} {self.score} with {self._elapsed_time}>"


class ExecutionFileIO:

    def __init__(self, f):
        self._file_handler = f

    def write(self, execution: ExecutionIteration):
        import base64
        pickled = base64.b64encode(pickle.dumps(execution)).decode()
        self._file_handler.write(pickled)
        self._file_handler.write('\n')

    def read_records(self):
        import base64
        for line in self._file_handler:
            yield pickle.loads(base64.b64decode(line))


class RandomHyperOptimizer:

    def __init__(self, parameters: Dict[str, distributions.Distribution] = None):
        self.parameters: Dict[str, distributions.Distribution] = {}
        if parameters is not None:
            self.parameters = parameters
        self.history : List[ExecutionIteration] = []

    def next_execution(self) -> ExecutionIteration:
        """Get the next proposed execution"""

        config = {
            name: param.sample()
            for name, param in self.parameters.items()
        }

        exec = ExecutionIteration(config)
        self.history.append(exec)

        return exec


