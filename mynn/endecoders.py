from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np


class LabelEncoderDecoder(metaclass=ABCMeta):
    """
    A label encoder/decoder will encode before training and decode at prediction of output variable
    """

    @abstractmethod
    def encode(self, y_variable: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def decode(self, nn_output: np.ndarray):
        pass


class ThresholdBinaryEncoderDecoder(LabelEncoderDecoder):
    """
    A binary encoder/decoder that will use a simple threshold to split probability space in two classes.
    """

    def __init__(self, probability_threshold: float = 0.5, class_a: int = 1, class_b: int = 0):
        """
        Initialize encoder/decoder with specific class and probability
        :param probability_threshold: The threshold above which the class_a will be selected
        """
        self.probability_threshold = probability_threshold
        self.class_a = class_a
        self.class_b = class_b

        if self.class_a == self.class_b:
            raise ValueError("Class A and B must be different")

    def encode(self, y_variable: np.ndarray) -> np.ndarray:
        if self.class_b == 1 and self.class_b == 0:
            return y_variable  # No need to transform if classes are already normalized in probabilities

        return np.where(y_variable == self.class_a,
                        1.,
                        0.).reshape(1, -1)

    def decode(self, nn_output: np.ndarray) -> np.ndarray:
        return np.where(nn_output > self.probability_threshold,
                        self.class_a,
                        self.class_b).reshape(1, -1)


class OneHotEncoderDecoder(LabelEncoderDecoder):
    """
    Encoder/Decoder for probabilities of one-hot-encoded data. Usually used on a softmax output
    """

    def __init__(self, classes: List):
        """
        Initialize one-hot decoder
        :param classes: A list of class per vector index.
        """
        if not isinstance(classes, list):
            raise TypeError("Argument \"classes\" must be a list of values.")
        self.classes: np.ndarray = None if classes is None else np.array(classes)

    def encode(self, y_variable: np.ndarray) -> np.ndarray:
        found_classes = np.unique(y_variable)

        unknown_classes = set(found_classes) - set(self.classes)
        if unknown_classes:
            raise ValueError(f"Y variable contains the following unknown classes: \"{unknown_classes}")

        return np.concatenate(tuple(
            np.array(y_variable == cls_value, dtype='float')
            for cls_value in self.classes), axis=0)

    def decode(self, nn_output: np.ndarray) -> np.ndarray:

        indices = np.argmax(nn_output, axis=0)

        # Translate indices to classes
        if nn_output.shape[0] != len(self.classes):
            raise ValueError("Y probabilities are of d")

        return self.classes[indices].reshape(1, -1)

