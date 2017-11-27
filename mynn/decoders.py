from abc import ABCMeta, abstractmethod
from typing import List, Optional

import numpy as np


class LabelDecoder(metaclass=ABCMeta):
    """
    A label decoder will take the probability results and decode them to initial
    labels
    """

    @abstractmethod
    def predict(self, y_probabilities: np.ndarray):
        pass


class ThresholdBinaryDecoder(LabelDecoder):
    """
    A binary decoder that will use a simple threshold to split probability space in two classes.
    """

    def __init__(self, probability_threshold: float = 0.5, class_a: int = 1, class_b: int = 0):
        """
        Initialize decoder
        :param probability_threshold: The threshold above which the class_a will be selected
        """
        self.probability_threshold = probability_threshold
        self.class_a = class_a
        self.class_b = class_b

        if self.class_a == self.class_b:
            raise ValueError("Class A and B must be different")

    def predict(self, y_probabilities: np.ndarray) -> np.ndarray:
        return np.where(y_probabilities > self.probability_threshold,
                        self.class_a,
                        self.class_b).reshape(-1, 1)


class OneHotDecoder(LabelDecoder):
    """
    Decoder for probabilities of one-hot-encoded data. Usually used on a softmax output
    """

    def __init__(self, classes: Optional[List] = None):
        """
        Initialize one-hot decoder
        :param classes: A list of class per vector index. If None the absolute value of index will be
        used.
        """
        if classes is not None and not isinstance(classes, list):
            raise TypeError("Argument \"classes\" must be a list of values.")
        self.classes = None if classes is None else np.array(classes)

    def predict(self, y_probabilities: np.ndarray) -> np.ndarray:

        indices = np.argmax(y_probabilities, axis=0)
        if self.classes is None:
            return indices.reshape(-1, 1)

        # Translate indices to classes
        if y_probabilities.shape[0] != len(self.classes):
            raise ValueError("Y probabilities are of d")

        return self.classes[indices].reshape(-1, 1)
