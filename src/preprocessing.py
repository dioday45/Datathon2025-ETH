import pickle
import re
from abc import ABC

import holidays
import pandas as pd


class PreProcessClass(ABC):
    def __init__(self, x: pd.DataFrame, features: pd.DataFrame):
        x.index = pd.to_datetime(x.index)

        features.index = pd.to_datetime(features.index)
        features = features[~features.index.duplicated(keep="first")]
        self.x = pd.concat([x, features], axis=1, join="inner")

        model_path = "model.pkl"

        # Load the model
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def preprocess_nonan(self, id: str) -> pd.DataFrame:
        """
        Extracts and cleans the time series for the given customer ID.
        """

        if id not in self.x.columns:
            raise ValueError(f"Customer ID '{id}' not found in the dataset.")

        if "ES" in id:
            # Use Spanish holidays
            country_holidays = holidays.country_holidays("ES")
        elif "IT" in id:
            # Use Italian holidays
            country_holidays = holidays.country_holidays("IT")
        else:
            # Default to Spanish holidays if the country is not recognized
            raise ValueError(f"Country not recognized for ID '{id}'")

        customer_ts = self.x[[id, "spv", "temp"]]
        customer_ts = customer_ts.rename(columns={id: "Consumption"})
        first_idx = customer_ts["Consumption"].first_valid_index()
        customer_ts = customer_ts.loc[first_idx:]
        customer_ts.index = pd.to_datetime(customer_ts.index)

        customer_ts = self.preprocess(customer_ts, id)

        # Add additional features
        customer_ts["Hour"] = customer_ts.index.hour.astype(int)
        customer_ts["Day"] = customer_ts.index.day.astype(int)
        customer_ts["Month"] = customer_ts.index.month.astype(int)
        customer_ts["Year"] = customer_ts.index.year.astype(int)
        customer_ts["Dow"] = customer_ts.index.day_of_week.astype(int)
        customer_ts["IsWeekend"] = (customer_ts.index.weekday >= 5).astype(int)
        customer_ts["is_holiday"] = [
            int(d in country_holidays) for d in customer_ts.index.normalize().date
        ]

        # # Create special_weekend feature
        # customer_ts["IsWeekendSpecial"] = 0
        # customer_ts.loc[
        #     (customer_ts["Dow"] == 5) & (customer_ts["Hour"] >= 20),
        #     "IsWeekendSpecial",
        # ] = 1
        # customer_ts.loc[(customer_ts["Dow"] == 6), "IsWeekendSpecial"] = 1
        # customer_ts.loc[
        #     (customer_ts["Dow"] == 0) & (customer_ts["Hour"] <= 6),
        #     "IsWeekendSpecial",
        # ] = 1

        # Create active_day feature
        customer_ts["ActiveDay"] = 0
        customer_ts.loc[
            (customer_ts["Hour"] >= 6) & (customer_ts["Hour"] <= 20), "ActiveDay"
        ] = 1

        customer_ts["Consumption"] = customer_ts["Consumption"].clip(
            lower=customer_ts["Consumption"].quantile(0.01),
            upper=customer_ts["Consumption"].quantile(0.99),
        )

        return customer_ts

    def preprocess(self, x: pd.DataFrame, id) -> pd.DataFrame:
        """data cleaning + imputation + standardization etc."""

        # Path to your model file
        temporary_df = x.copy()
        id_short = re.search(r"(IT|ES)_\d+", id).group()
        temporary_df["number"] = id_short
        temporary_df["hour"] = temporary_df.index.hour
        temporary_df["day_of_week"] = temporary_df.index.dayofweek
        temporary_df["month"] = temporary_df.index.month
        temporary_df["year"] = temporary_df.index.year

        temporary_df = temporary_df[["number", "hour", "day_of_week", "month", "year"]]

        temporary_df.number = temporary_df.number.astype("category")

        prediction = self.model.predict(
            temporary_df, num_iteration=self.model.best_iteration
        )

        x.loc[x["Consumption"].isna(), "Consumption"] = prediction[
            x["Consumption"].isna()
        ]

        return x
