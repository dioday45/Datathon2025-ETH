import pickle
import re
from abc import ABC

import holidays
import numpy as np
import pandas as pd


class PreProcessClass(ABC):
    def __init__(self, x: pd.DataFrame, features: pd.DataFrame):
        x.index = pd.to_datetime(x.index)

        features.index = pd.to_datetime(features.index)
        features = features[~features.index.duplicated(keep="first")]
        self.x = pd.concat([x, features], axis=1, join="outer")
        self.x = self.x[self.x.index < pd.to_datetime("2024-09-01")]
        model_path = "model.pkl"

        # Load the model
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def preprocess_nonan(self, id: str) -> pd.DataFrame:
        """
        Extracts and cleans the time series for the given customer ID.
        """

        for i in id:
            if i not in self.x.columns:
                raise ValueError(f"Customer ID '{i}' not found in the dataset.")

        if len(id) == 1:
            if "ES" in id[0]:
                # Use Spanish holidays
                country_holidays = holidays.country_holidays("ES")
            elif "IT" in id[0]:
                # Use Italian holidays
                country_holidays = holidays.country_holidays("IT")
            else:
                # Default to Spanish holidays if the country is not recognized
                raise ValueError(f"Country not recognized for ID '{id}'")

        customer_ts = self.x[id + ["spv"] + ["temp"]]
        if len(id) == 1:
            customer_ts = customer_ts.rename(columns={id[0]: "Consumption"})
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
        if len(id) == 1:
            customer_ts["is_holiday"] = [
                int(d in country_holidays) for d in customer_ts.index.normalize().date
            ]
        customer_ts["hour_sin"] = np.sin(2 * np.pi * customer_ts["Hour"] / 24)
        customer_ts["hour_cos"] = np.cos(2 * np.pi * customer_ts["Hour"] / 24)

        customer_ts["dow_sin"] = np.sin(2 * np.pi * customer_ts["Dow"] / 7)
        customer_ts["dow_cos"] = np.cos(2 * np.pi * customer_ts["Dow"] / 7)

        customer_ts["month_sin"] = np.sin(2 * np.pi * customer_ts["Month"] / 12)
        customer_ts["month_cos"] = np.cos(2 * np.pi * customer_ts["Month"] / 12)

        if len(id) == 1:
            customer_ts = customer_ts.drop(
                columns=["Hour", "Day", "Month", "Year", "Dow"]
            )

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
        # customer_ts["ActiveDay"] = 0
        # customer_ts.loc[
        #     (customer_ts["Hour"] >= 6) & (customer_ts["Hour"] <= 20), "ActiveDay"
        # ] = 1

        return customer_ts

    def preprocess(self, x: pd.DataFrame, id) -> pd.DataFrame:
        """data cleaning + imputation + standardization etc."""

        # Path to your model file
        temporary_df = x.copy()
        temporary_df["hour"] = temporary_df.index.hour
        temporary_df["day_of_week"] = temporary_df.index.dayofweek
        temporary_df["month"] = temporary_df.index.month
        temporary_df["year"] = temporary_df.index.year

        if len(id) == 1:
            pattern = r"(IT|ES)_\d+"

            id_short = re.search(r"(IT|ES)_\d+", id[0]).group()
            temporary_df["number"] = id_short

            temporary_df = temporary_df[
                ["number", "hour", "day_of_week", "month", "year"]
            ]

            temporary_df.number = temporary_df.number.astype("category")

            # Filter for rows where consumption data is missing
            missing_mask = x["Consumption"].isna()

            # Only proceed if there are missing values for the consumer
            if missing_mask.any():
                # Prepare a temporary DataFrame for prediction on missing values
                prediction_df = temporary_df[missing_mask][
                    ["number", "hour", "day_of_week", "month", "year"]
                ]

                # Perform prediction only on the rows where consumption is missing
                prediction = self.model.predict(
                    prediction_df, num_iteration=self.model.best_iteration
                )

                # Update the original DataFrame with the predicted values
                x.loc[x["Consumption"].isna(), "Consumption"] = prediction
            return x

        for consumer in x.columns:
            # print(consumer)
            pattern = r"(IT|ES)_\d+"

            if not re.search(pattern, consumer):
                continue
            id_short = re.search(r"(IT|ES)_\d+", consumer).group()
            temporary_df["number"] = id_short

            temporary_df = temporary_df[
                ["number", "hour", "day_of_week", "month", "year"]
            ]

            temporary_df.number = temporary_df.number.astype("category")

            # Filter for rows where consumption data is missing
            missing_mask = x[consumer].isna()

            # Only proceed if there are missing values for the consumer
            if missing_mask.any():
                # Prepare a temporary DataFrame for prediction on missing values
                prediction_df = temporary_df[missing_mask][
                    ["number", "hour", "day_of_week", "month", "year"]
                ]

                # Perform prediction only on the rows where consumption is missing
                prediction = self.model.predict(
                    prediction_df, num_iteration=self.model.best_iteration
                )

                x.loc[missing_mask, consumer] = prediction

        return x

    def preprocess_EDA(self, id: list[str]) -> pd.DataFrame:
                """
                Extracts and cleans the time series for the given customer ID.
                """

                for i in id:
                    if i not in self.x.columns:
                        raise ValueError(f"Customer ID '{i}' not found in the dataset.")

                customer_ts = self.x[id + ["spv"] + ["temp"]]
                customer_ts.index = pd.to_datetime(customer_ts.index)

                # Add additional features
                customer_ts["Hour"] = customer_ts.index.hour
                customer_ts["Day"] = customer_ts.index.day   
                customer_ts["Month"] = customer_ts.index.month
                customer_ts["Year"] = customer_ts.index.year
                customer_ts['Dow'] = customer_ts.index.day_name()
                customer_ts["DayYear"] = customer_ts.index.dayofyear
                customer_ts["Week"] = customer_ts.index.isocalendar().week
                customer_ts["Season"] = customer_ts.index.quarter
                customer_ts["IsWeekend"] = customer_ts.index.weekday >= 5

                # Create special_weekend feature
                customer_ts["IsWeekendSpecial"] = False
                customer_ts.loc[(customer_ts["Dow"] == "Saturday") & (customer_ts["Hour"] >= 20), "IsWeekendSpecial"] = True
                customer_ts.loc[(customer_ts["Dow"] == "Sunday"), "IsWeekendSpecial"] = True
                customer_ts.loc[(customer_ts["Dow"] == "Monday") & (customer_ts["Hour"] <= 6), "IsWeekendSpecial"] = True

                # Create active_day feature
                customer_ts["ActiveDay"] = False
                customer_ts.loc[(customer_ts["Hour"] >= 6) & (customer_ts["Hour"] <= 20), "ActiveDay"] = True
                
                # Rename season
                season_map = {
                    1: "Winter",
                    2: "Spring",
                    3: "Summer",
                    4: "Autumn"
                }
                
                customer_ts["Season"] = customer_ts["Season"].map(season_map)

                list_to_cat = [
                    "Hour",
                    "Day",
                    "Month",
                    "Year",
                    "Dow",
                    "DayYear",
                    "Week",
                    "Season"
                ]

                list_to_bool = [
                    "IsWeekendSpecial",
                    "ActiveDay"
                ]

                customer_ts[list_to_cat] = customer_ts[list_to_cat].astype("category")
                customer_ts[list_to_bool] = customer_ts[list_to_bool].astype("bool")

                return customer_ts