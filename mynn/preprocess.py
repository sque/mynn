from typing import Optional
import numpy as np


class StandardScaler:
    """
    Scale features by removing the mean and scaling to unit variance
    """

    def __init__(self, mean: Optional[float] = None, variance: Optional[float] = None) -> None:
        """
        Initialize scaler
        :param mean: This is the mean of the original data
        :param variance: This is the variance of the original data
        """
        self._mean = mean
        self._variance = variance

    def train(self, X: np.ndarray) -> None:
        """
        Train scaler on a dataset. This will estimate the mean and variance of the dataset
        :param X: The dataset to train scaler
        """
        self._mean = np.mean(X, axis=1, keepdims=True)
        self._variance = np.var(X, axis=1, keepdims=True)

    def scale(self, X: np.ndarray) -> np.ndarray:
        """
        Scale a dataset based on the mean and variance that was previously trained.
        :param X: The dataset to scale
        :return: The scaled version of the dataset
        """
        return (X - self._mean) / self._variance

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self._mean}, variance={self._variance})"
