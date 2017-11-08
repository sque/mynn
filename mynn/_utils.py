from typing import Iterable, Tuple
from itertools import zip_longest, chain


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
