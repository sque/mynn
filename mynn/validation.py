from typing import Dict, Tuple, List, Optional, Any

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from mynn.loss import CrossEntropyLoss
from mynn.network import FNN
from mynn._utils import TrainingContext


def prediction_performance(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary') -> Dict[str, np.ndarray]:
    """
    Calculate multiple prediction performance scores for a given dataset.

    The calculated scores are: `accuracy, `precision`, `recall`, `f1`, `cross_entropy`

    :param y_true: The ground truth dataset
    :param y_pred: The prediction output
    :param average: This parameter is required for multiclass/multilabel targets in `precision`, `recall`,
    `f1`, `cross_entropy`
    :return: A set of metrics mapped by their friendly name.
    """
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average),
        'recall': recall_score(y_true, y_pred, average=average),
        'f1': f1_score(y_true, y_pred, average=average),
        'cross_entropy': CrossEntropyLoss()(y_pred, y_true)
    }


def performance_training_callback(datasets_x: List[np.ndarray], datasets_y: List[np.ndarray],
                                      every_nth_iteration=100,
                                      prediction_kwargs: Optional[Dict[str, Any]] = None):
    """
    Training callback to generate extra performance metrics against multiple datasets.
    :param datasets_x: A list of input features for each dataset
    :param datasets_y: A list of the output variables for each dataset
    :param every_nth_iteration: The period of training iterations for which the stats will be calculated
    :param prediction_kwargs: Arguments to set for prediction_performance
    :return: The callback to be used by FNN.train()
    """

    prediction_kwargs = prediction_kwargs or {}

    if len(datasets_x) != len(datasets_y):
        raise ValueError("Datasets X and Y must be of the same size.")

    def _post_iteration_callback(nn: FNN, ctx: TrainingContext) -> bool:
        if ctx.current_iteration_index % every_nth_iteration != 0:
            return False

        for dataset_idx, (dataset_x, dataset_y) in enumerate(zip(datasets_x, datasets_y)):
            stats = prediction_performance(dataset_y, nn.predict(dataset_x), **prediction_kwargs)
            for stat_name, value in stats.items():
                ctx.report_cost(f"{stat_name}_{dataset_idx}", value)

    return _post_iteration_callback
