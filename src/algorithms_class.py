from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class TimeSeriesPrediction(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, x_train: pd.DataFrame, y: pd.DataFrame) -> None:
        pass

    def loss(
        self, y_pred: pd.Series, y_true: pd.Series, w1: float = 1, w2: float = 1
    ) -> float:
        """TBD.

        Inputs:
            y_pred/y_true: (n,) predictions
            w1/w2: weights for loss computation
        """
        pass
