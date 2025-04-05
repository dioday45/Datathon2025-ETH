from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

from tqdm import tqdm


class Model(ABC):
    def __init__(self):
        super().__init__(ABC)

    @abstractmethod
    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        """simple fit data."""
        pass

    @abstractmethod
    def train(self, x: pd.DataFrame, y: pd.DataFrame, split: TimeSeriesSplit) -> None:
        """split train."""
        pass

    @abstractmethod
    def predict(self, x_test: pd.DataFrame) -> pd.DataFrame:
        pass

    def loss_porfolio_level(self, y_pred: pd.DataFrame, y_true: pd.Series) -> pd.Series:
        """compute the loss of the predictions at PORTFOLIO level.

        Input:
            y_pred: (n, number_of_clients) predictions of each client, a column is a client, a row is an hour
            y_true: (n,) observations of consumption of portfolio
        Output:
            loss: (n,) losses at each hour
        """
        sum_consumption_pred = y_pred.sum(axis=1)
        return np.abs(sum_consumption_pred - y_true)

    def loss_client_level(
        self, y_pred: pd.DataFrame, y_true: pd.DataFrame
    ) -> pd.DataFrame:
        """copute the loss of predictions at CLIENT level.

        Input:
            y_pred: (n, number_of_clients) predictions of each client, a column is a client, a row is an hour
            y_true: (n, number_of_clients) observations of consumption by client
        Output:
            loss: (n, number_of_clients) losses at each hour for each client
        """
        return np.abs(y_pred - y_true).sum(axis=1)

    def loss(
        self,
        y_pred_client: pd.DataFrame,
        y_true_client: pd.DataFrame,
        y_pred_portfolio: pd.Series,
        y_true_portfolio: pd.Series,
        w1: float = 1,
        w2: float = 1,
    ) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
        """TBD.

        Inputs:
            y_pred/y_true: see def of losses
            w1/w2: weights for loss computation
        Output:
            total loss
            loss_clients
            loss_portfolio
        """
        loss_clients = self.loss_client_level(y_pred_client, y_true_client)
        loss_portfolio = self.loss_porfolio_level(y_pred_portfolio, y_true_portfolio)
        total_loss = w1 * loss_clients + w2 * loss_portfolio

        return total_loss, loss_clients, loss_portfolio


class SimpleModel(Model):
    """
    Baseline model (regression).
    """

    def __init__(self):
        super().__init__(Model)
        self.linear_regression = LinearRegression()

    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        self.linear_regression.fit(x, y)

    def train(
        self, x: pd.DataFrame, y: pd.DataFrame, split: TimeSeriesSplit
    ) -> tuple[list]:
        """cross validation train loop.

        Input:
            split:  define a timeseries split (CV)
        """
        losses_train = []
        losses_eval = []

        for _, (train_index, eval_index) in tqdm(enumerate(split.split(x))):
            x_train = x[train_index]
            y_train = y[train_index]
            x_eval = x[eval_index]
            y_eval = y[eval_index]

            self.fit(x_train, y_train)
            y_train_pred = self.predict(x_train)
            y_eval_pred = self.predict(x_eval)

            loss_train, *_ = self.loss(y_train_pred, y_train)
            loss_eval, *_ = self.loss(y_eval_pred, y_eval)

            losses_train.append(loss_train)
            losses_eval.append(loss_eval)

        return losses_train, losses_eval

    def predict(self, x):
        return self.linear_regression.predict(x)
