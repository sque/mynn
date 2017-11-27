from typing import Callable


def approximated_derivative(f: Callable, *points, e=0.1e-8):
    """
    Approximate derivative
    :param f: The function to calculate derivative
    :param points: The point of the function f to calculate derivative
    :param e: The
    """
    min_points = list(map(lambda x: x - e, points))
    max_points = list(map(lambda x: x + e, points))
    return (f(*max_points) - f(*min_points)) / (2*e)

