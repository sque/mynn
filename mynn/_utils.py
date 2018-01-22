from typing import Iterable, Tuple, Generator, Optional, List
from itertools import zip_longest, chain
import pandas as pd
import numpy as np
import logging as _logging
import time
from datetime import timedelta, datetime

logger = _logging.getLogger(__name__)


def grouped(s: Iterable, n: int) -> Iterable[Tuple]:
    """
    Given a flatten array return a list of groups of size `n`

    s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ...

    :param s: An iterable sequence
    :param n:
    :return:
    """
    # Create n positional iterators that point to the same iterator
    positional_iterators = [iter(s)] * n

    # Zip them together in order to be iterated consequently by zip,
    # This will give the effect of never reusing an item more than one time.
    return zip_longest(*positional_iterators)


def nested_chain_iterable(s: Iterable, levels: int) -> Tuple:
    """
    Perform nested chain on an iterable in order to flatten a multi-level
    structure.
    :param s: A sequence of other sequences
    :param levels: The level to flatten iterables
    :return: An chain iterator for n-level items
    """

    iterator = chain.from_iterable(s)
    for i in range(levels - 1):
        iterator = chain.from_iterable(iterator)

    return iterator


def mini_batches(X: np.ndarray, Y: np.ndarray, mini_batch_size: Optional[int]) -> Generator[np.ndarray, None, None]:
    """
    Generator for mini-batches that are guaranteed to return all samples only one time.
    Each mini-batch will be created from a shuffled pool of samples
    :param X: The input X array
    :param Y: The input Y array
    :param mini_batch_size: The size of each mini-batch
    :return: The X and Y for each mini-batch
    """
    total_samples = X.shape[1]

    if mini_batch_size is None or total_samples <= mini_batch_size:
        yield X, Y
        return

    # Shuffle them
    shuffled_indices = np.random.permutation(total_samples)

    logger.debug(f"Shuffling dataset for improved optimization performance")
    X = X[:, shuffled_indices]
    Y = Y[:, shuffled_indices]

    for batch_start in range(0, total_samples, mini_batch_size):
        batch_end = batch_start + min(batch_start + mini_batch_size, total_samples)
        X_batch = X[:, batch_start:batch_end]
        Y_batch = Y[:, batch_start:batch_end]
        yield X_batch, Y_batch


def random_mini_batches(X: np.ndarray, Y: np.ndarray, mini_batch_size: Optional[int]) -> \
        Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Create mini-batches by randomly selecting samples from dataset in each iteration. This algorithm
    does not guarantees to visit all the batches nor that each sample will be in only one sample.
    Each mini-batch will be created from a shuffled pool of samples
    :param X: The input X array
    :param Y: The input Y array
    :param mini_batch_size: The size of each mini-batch
    :return: The X and Y for each mini-batch
    """

    total_samples = X.shape[-1]

    if mini_batch_size is None or total_samples <= mini_batch_size:
        yield X, Y
        return

    # Find the appropriate size of batch sizes
    batch_sizes = [mini_batch_size] * (total_samples // mini_batch_size)
    if total_samples % mini_batch_size:
        batch_sizes.append(total_samples % mini_batch_size)

    # Return each batch by randomly selecting from dataset
    for current_mini_batch_size in batch_sizes:
        shuffled_indices = np.random.permutation(total_samples)[:current_mini_batch_size]
        yield X[..., shuffled_indices], Y[:, shuffled_indices]


class RelativeTime:
    """
    Helper class to keep tha relative time since the beginning
    """

    def __init__(self):
        self._start: float = None
        self.reset()

    def reset(self):
        """Reset the timer"""
        self._start = time.time()

    @property
    def passed_time(self) -> float:
        """
        Get the relative time since the bigning
        """
        return time.time() - self._start

    @property
    def passed_timedelta(self) -> timedelta:
        """
        Get the relative time since the beginning
        """
        return timedelta(seconds=self.passed_time)


class TrainingIteration:
    """
    Wrapper for record data of a training iteration
    """

    def __init__(self, finish_time: datetime, epoch=None, mini_batch_index=None, iteration_index=None):
        """
        Initialize record data of training iteration
        :param finish_time: The time it took to finish this iteraiton
        :param epoch: The epoch of this iteration
        :param mini_batch_index: The batch size
        :param iteration_index: The absolute index of iteration since the beginning of the training
        """
        self.finish_time = finish_time
        self.epoch = epoch
        self.mini_batch_index = mini_batch_index
        self.iteration_index = iteration_index
        self.costs = {}

    def report_cost(self, name, value):
        """
        Report a cost of this iteration.
        The function is metric-agnostic and any type of scalar cost can be used.
        :param name: The unique name of the cost
        :param value: The value of the cost
        """
        if name in self.costs:
            raise KeyError(f"There is already a record for \"{name}\" cost.")
        self.costs[name] = value


class TrainingContext:
    """
    Training context for neural networks
    """

    def __init__(self):
        """
        Initialize context
        """

        self._current_epoch: int = None
        self._current_mini_batch_index: int = None
        self._current_iteration_index: int = 0

        self._iterations_history: List[TrainingIteration] = []

        # Timers
        self._start_timer = RelativeTime()
        self._epoch_timer = RelativeTime()

        # Average training time for each epoch
        self._epoch_uncorrected_average_time: float = 0
        self._epoch_average_time: float = 0

        self._total_scheduled_epochs = None
        self.reset()

    def reset(self):
        """
        Reset counters to start a new training
        """
        self._current_epoch = -1
        self._current_mini_batch_index = -1
        self._current_iteration_index = -1
        self._epoch_uncorrected_average_time = 0
        self._epoch_average_time = 0
        self._start_timer.reset()

    def iter_epochs(self, epochs: int) -> Generator[int, None, None]:
        """
        Get an iterator for epochs
        :param epochs: The number of epochs to iterate
        :return: A generator of epochs that will keep track of the current epoch and timers
        """
        self._total_scheduled_epochs = epochs
        for self._current_epoch in range(self._current_epoch + 1, self._current_epoch + epochs + 1):
            self._current_mini_batch_index = -1
            self._epoch_timer.reset()
            yield self._current_epoch

            # Weighted exponential average of the epoch time with bias correction
            current_time = self._epoch_timer.passed_timedelta.total_seconds()
            beta = 0.9

            self._epoch_uncorrected_average_time = (self._epoch_uncorrected_average_time * beta
                                                    + (1.0 - beta) * current_time)
            self._epoch_average_time = self._epoch_uncorrected_average_time \
                                       / (1 - beta ** (self._current_epoch + 1))  # Bias correction

    def iter_mini_batch(self, batch_generator: Generator[Tuple[np.ndarray, np.ndarray], None, None]) \
            -> Generator[int, None, None]:
        """
        Iterator for mini-batches
        :param batch_generator: A generator of mini-batches dataset
        :return: A generator that will keep track of the current mini-batch
        """
        for self._current_mini_batch_index, batch in enumerate(batch_generator, start=0):
            self._current_iteration_index += 1
            yield batch

    @property
    def total_scheduled_epochs(self) -> int:
        """The total number of scheduled epochs"""
        return self._total_scheduled_epochs

    @property
    def current_iteration_index(self) -> int:
        """The current iteration where iteration is the most nested loop in training."""
        return self._current_iteration_index

    @property
    def current_epoch(self) -> int:
        """The current executed epoch."""
        return self._current_epoch

    @property
    def current_mini_batch_index(self) -> int:
        """The index of the current mini-batch that is trained"""
        return self._current_mini_batch_index

    @property
    def passed_timedelta(self) -> timedelta:
        """Total time passed since the begining of the training"""
        return self._start_timer.passed_timedelta

    def iteration_done(self, loss: float) -> TrainingIteration:
        """
        Mark an iteration as finished
        :param loss: The final loss of the network at the end of this iteration.
        This must be the same loss metric as the one used to optimize the network
        :return: The iteration wrapper
        """
        training_iteration = TrainingIteration(
            datetime.now(),
            epoch=self.current_epoch,
            mini_batch_index=self.current_mini_batch_index,
            iteration_index=self.current_iteration_index)
        training_iteration.report_cost('loss', loss)
        self._iterations_history.append(training_iteration)
        return training_iteration

    @property
    def iterations_history(self) -> List[TrainingIteration]:
        """Get the list of all iterations performed in this training context"""
        return self._iterations_history

    def current_iteration(self) -> TrainingIteration:
        """Get record data of current iteration"""
        return self._iterations_history[-1]

    @property
    def estimated_remaining_time(self) -> Optional[timedelta]:
        """
        Estimate how much time is left for the training process to finish
        :return: If the time cannot be estimated it will return None
        """
        if not self._epoch_average_time:
            return None

        left_seconds = (self.total_scheduled_epochs - self.current_epoch) * self._epoch_average_time
        return timedelta(seconds=left_seconds)

    def report_cost(self, name: str, value: float) -> None:
        """
        Report cost for the current iteration
        :param name: The name of the cost metric
        :param value: The value of the metric
        """
        self._iterations_history[-1].report_cost(name, value)

    def iterations_history_as_dataframe(self) -> pd.DataFrame:
        """
        Get all iterations meta-data as pandas dataframe
        """

        def _merge_dict(*dicts):
            merged = dicts[0].copy()
            for d in dicts[1:]:
                merged.update(d)
            return merged

        data = [_merge_dict(

            {
                attr: getattr(h, attr)
                for attr in ['finish_time', 'epoch', 'mini_batch_index', 'iteration_index']
            },
            h.costs
        )

            for h in self._iterations_history]
        return pd.DataFrame(data)
