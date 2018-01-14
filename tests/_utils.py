import numpy as np
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
    return (f(*max_points) - f(*min_points)) / (2 * e)


def approximated_derivative_parameters(f: Callable, *parameters, e=0.1e-8):
    """
    Approximate derivatives of multiple parameters model
    :param f: The function to calculate derivative
    :param points: The point of the function f to calculate derivative
    :param e: The
    """

    def copy_parameters():

        return [
            p.copy()
            for p in parameters
        ]

    d_parameters = [
        np.empty((p.shape))
        for p in parameters
    ]

    for pi, p in enumerate(parameters):
        for vi in range(p.size):
            sample_value = p.flat[vi]
            min_params = copy_parameters()
            min_params[pi].flat[vi] = sample_value - e

            max_params = copy_parameters()
            max_params[pi].flat[vi] = sample_value + e
            # print(max_params)
            # print(min_params)
            dparam = (f(*max_params) - f(*min_params)) / (2 * e)
            d_parameters[pi].flat[vi] = dparam
    return d_parameters
