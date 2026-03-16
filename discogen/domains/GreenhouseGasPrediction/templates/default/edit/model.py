from typing import Protocol

import numpy as np
from sklearn.linear_model import LinearRegression
from typing_extensions import Self

class Model(Protocol):
    predicts_std: bool
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        pass


def make_model(pre_processed_data: np.ndarray) -> Model:
    """
    Make a model from the pre-processed data.

    Args:
        pre_processed_data (N, 5): The pre-processed data.

    Returns:
        Model: The model.
    """

    # EDIT HERE
    model = ...
    return model
