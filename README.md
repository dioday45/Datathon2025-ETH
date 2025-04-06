# Electricity Demand Forecasting Project
**Hackathon 2025 with Alpiq**

## Project Overview

This project focuses on predicting hourly electricity demand for thousands of consumers from Italy and Spain. During the 2025 Alpiq Hackathon 2025 the teams had implemented a baseline predictive model of energy demand based on the mean for a given day of the week and hour, using data from the previous month. Although the baseline model performed with a test set loss of **323k** for July 2024, more sophisticated models were developed to improve accuracy.

## Developed Models

- **Global Model**
  This model used the consumer ID as an additional input and helped impute missing (NaN) values between real demand data points.

- **Individual Consumer Models**
  Separate models were trained for each consumer. The feature set included:
  - Cyclic period transformations (sine/cosine)
  - Temperature
  - Sunlight
  - Weekend flags
  - Holidays
  - Lag features (1 day, 1 week, 2 weeks)

  These features were selected after careful exploratory data analysis (EDA).

The chosen model for both imputation and forecasting was **LightGBM**, a gradient boosting framework designed for speed and efficiency.

## Repository Structure

- `README.md`: Contains the project overview and setup instructions.
- `datathon2025_quAIntly_presentation.ppxt`: PowerPoint presentation of the project
- `configs/`: Configuration files related to the project.
- `datasets2025/`: Contains data files for various regions (e.g., Italy, Spain), including historical metering data, holiday data, and forecasts.
- `environmentAlpiqDatathon.yml`: The environment setup file for the project dependencies.
- `model.pkl`: The trained model saved for later use.
- `notebooks/`: Jupyter notebooks for exploration, model building, and analysis, including specific notebooks for individual tasks like clustering and per-consumer modeling.
- `predictions/`: Contains the output of model predictions for Italy and Spain.
- `scripts/`: Python scripts for data processing, forecasting, and scoring.
- `setup.py`: Setup script for packaging the project.
- `src/`: Contains the core Python code for data processing, feature engineering, model training, and evaluation.

## How-to
How to use this repo? First, install the dependencies by:
```
conda create -f environmentAlpiqDatathon.yml
pre-commit install
```
The dataset needs to be in the root. Then, load the data with the `DataLoader` class and pass the observations in the `Preprocess` class. Call the `preprocess` function that preprocesses the data (mainly data imputation with the General Model, whose weights are zipped in the root -and need to be unzipped). The notebook `Model_for_imputation` contains... the model fitting for imputing the missing data. You can find the EDA in the `EDA.ipynb` notebook. The main notebook, `per_consumer_modeling.ipynb`, contains the preceding, with a feature analysis and modeling predictions.


## Acknowledgments

Thanks to **Alpiq** for the interesting challenge and the opportunity to tackle such a complex forecasting problem!
