import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit


def train_test_split_data(x: pd.DataFrame, y: pd.DataFrame, kfolds: int = 5):
    pass


def drop_before(ts: pd.Series):
    first_nan = ts.first_valid_index()
    return ts[first_nan:]


def find_nan_streaks(ts: pd.Series):
    nan_streak_starts = ts.isna() & ~ts.isna().shift(fill_value=False)
    nan_streak_ends = ts.isna() & ~ts.isna().shift(-1, fill_value=False)
    first_nan_indices = ts.index[nan_streak_starts]
    last_nan_indices = ts.index[nan_streak_ends]

    return last_nan_indices - first_nan_indices, first_nan_indices
