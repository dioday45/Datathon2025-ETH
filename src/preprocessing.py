from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class PreProcessClass(ABC):
    def __init__(self, x: pd.DataFrame):
        self.x = x

    def preprocess_nonan(self, id: pd.DataFrame) -> pd.DataFrame:
        """data cleaning + imputation + standardization etc."""
        # fill na
        customer_ts = self.x[id].dropna()

        return customer_ts

    def preprocess(self, x: pd.DataFrame) -> pd.DataFrame:
        """data cleaning + imputation + standardization etc."""
        # fill na

        # features
        pass

    # @abstractmethod
    # def feature_engineering(self, x: pd.DataFrame) -> pd.DataFrame:
    #     pass
