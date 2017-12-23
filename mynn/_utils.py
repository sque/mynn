from typing import Iterable, Tuple, Generator, Optional
from itertools import zip_longest, chain
import numpy as np
import logging as _logging
import time
from datetime import timedelta


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
    for i in range(levels-1):
        iterator = chain.from_iterable(iterator)

    return iterator


def mini_batches(X:np.ndarray, Y:np.ndarray, mini_batch_size:Optional[int]) -> Generator[np.ndarray, None, None]:
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


def random_mini_batches(X:np.ndarray, Y:np.ndarray, mini_batch_size:Optional[int]) -> Generator[np.ndarray, None, None]:
    """
    Create mini-batches by randomly selecting samples from dataset in each iteration. This algorithm
    does not guarantees to visit all the batches nor that each sample will be in only one sample.
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

    # Find the appropriate size of batch sizes
    batch_sizes = [mini_batch_size] * (total_samples // mini_batch_size)
    if total_samples % mini_batch_size:
        batch_sizes.append(total_samples % mini_batch_size)

    # Return each batch by randomly selecting from dataset
    for current_mini_batch_size in batch_sizes:
        shuffled_indices = np.random.permutation(total_samples)[:current_mini_batch_size]
        # print(shuffled_indices)
        yield X[:, shuffled_indices], Y[:, shuffled_indices]


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
        Get the relative time since the bigning
        """
        return timedelta(seconds=self.passed_time)
