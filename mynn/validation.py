from typing import Dict, Tuple, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from mynn.loss import CrossEntropyLoss
from mynn.network import FNN


def prediction_performance(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calculate multiple prediction performance scores for a given dataset.

    The calculated scores are: `accuracy, `precision`, `recall`, `f1`, `cross_entropy`

    :param y_true: The ground truth dataset
    :param y_pred: The prediction output
    :return: A set of metrics mapped by their friendly name.
    """
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'cross_entropy': CrossEntropyLoss()(y_pred, y_true)
    }


def performance_training_callback(datasets_x: List[np.ndarray], datasets_y: List[np.ndarray], every_nth_iteration=100):
    """
    Training callback to generate extra performance metrics against multiple datasets.
    :param datasets_x: A list of input features for each dataset
    :param datasets_y: A list of the output variables for each dataset
    :param every_nth_iteration: The period of training iterations for which the stats will be calculated
    :return: The callback to be used by FNN.train()
    """

    if len(datasets_x) != len(datasets_y):
        raise ValueError("Datasets X and Y must be of the same size.")

    def _iteration_callback(nn: FNN, epoch:int, mini_batch:int, iteration:int) -> Tuple[Optional[Dict], ...]:
        if iteration % every_nth_iteration != 0:
            return (None, ) * len(datasets_x)

        stats = tuple(
            prediction_performance(dataset_y, nn.predict(dataset_x))
            for dataset_x, dataset_y in zip(datasets_x, datasets_y)
        )

        return stats

    return _iteration_callback
