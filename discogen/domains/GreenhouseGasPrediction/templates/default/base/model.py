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


def make_model(data: np.ndarray) -> Model:

    class LinRegModel(LinearRegression):
        predicts_std = False

    return LinRegModel()
