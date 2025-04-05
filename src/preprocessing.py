from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class PreProcessClass(ABC):
    def __init__(self):
        pass

    def preprocess(self, x: pd.DataFrame):
        pass

    @abstractmethod
    def feature_engineering(self, x: pd.DataFrame):
        pass
