from abc import ABC, abstractmethod
from src.data import DataLoader

from os.path import join

import numpy as np
import pandas as pd

from src.utils import *


class PreProcessClass(ABC):
    def __init__(self, x: pd.DataFrame, features: pd.DataFrame):
        x.index = pd.to_datetime(x.index)

        features.index = pd.to_datetime(features.index)
        features = features[~features.index.duplicated(keep="first")]
        self.x = pd.concat([x, features], axis=1, join="inner")

    def preprocess_nonan(self, id: str) -> pd.DataFrame:
        """
        Extracts and cleans the time series for the given customer ID.
        """

        if id not in self.x.columns:
            raise ValueError(f"Customer ID '{id}' not found in the dataset.")

        def drop_before(ts: pd.Series):
            first_nan = ts.first_valid_index()
            return ts[first_nan:]

        def find_nan_streaks(ts: pd.Series):
            nan_streak_starts = ts.isna() & ~ts.isna().shift(fill_value=False)
            nan_streak_ends = ts.isna() & ~ts.isna().shift(-1, fill_value=False)
            first_nan_indices = ts.index[nan_streak_starts]
            last_nan_indices = ts.index[nan_streak_ends]

            return last_nan_indices - first_nan_indices, first_nan_indices

        customer_ts = self.x[[id, "spv", "temp"]].dropna(subset=[id])
        customer_ts = drop_before(customer_ts)
        length_nan, start_nan = find_nan_streaks(customer_ts[id])
        print(length_nan)
        print(start_nan)
        customer_ts = customer_ts.rename(columns={id: "Consumption"})

        if not pd.api.types.is_datetime64_any_dtype(customer_ts.index):
            customer_ts.index = pd.to_datetime(customer_ts.index)

        # Add additional features
        customer_ts["Hour"] = customer_ts.index.hour
        customer_ts["Day"] = customer_ts.index.day
        customer_ts["Month"] = customer_ts.index.month
        customer_ts["Year"] = customer_ts.index.year
        customer_ts["Dow"] = customer_ts.index.day_name()
        customer_ts["DayYear"] = customer_ts.index.dayofyear
        customer_ts["Week"] = customer_ts.index.isocalendar().week
        customer_ts["Season"] = customer_ts.index.quarter
        customer_ts["IsWeekend"] = customer_ts.index.weekday >= 5

        # Create special_weekend feature
        customer_ts["IsWeekendSpecial"] = False
        customer_ts.loc[
            (customer_ts["Dow"] == "Saturday") & (customer_ts["Hour"] >= 20),
            "IsWeekendSpecial",
        ] = True
        customer_ts.loc[(customer_ts["Dow"] == "Sunday"), "IsWeekendSpecial"] = True
        customer_ts.loc[
            (customer_ts["Dow"] == "Monday") & (customer_ts["Hour"] <= 6),
            "IsWeekendSpecial",
        ] = True

        # Create active_day feature
        customer_ts["ActiveDay"] = False
        customer_ts.loc[
            (customer_ts["Hour"] >= 6) & (customer_ts["Hour"] <= 20), "ActiveDay"
        ] = True

        # Rename season
        season_map = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Autumn"}

        customer_ts["Season"] = customer_ts["Season"].map(season_map)

        list_to_cat = [
            "Hour",
            "Day",
            "Month",
            "Year",
            "Dow",
            "DayYear",
            "Week",
            "Season",
        ]

        list_to_bool = ["IsWeekendSpecial", "ActiveDay"]

        customer_ts[list_to_cat] = customer_ts[list_to_cat].astype("category")
        customer_ts[list_to_bool] = customer_ts[list_to_bool].astype("bool")

        return customer_ts

    def preprocess(self, x: pd.DataFrame) -> pd.DataFrame:
        """data cleaning + imputation + standardization etc."""
        # fill na

        # features
        pass

    # @abstractmethod
    # def feature_engineering(self, x: pd.DataFrame) -> pd.DataFrame:
    #     pass
