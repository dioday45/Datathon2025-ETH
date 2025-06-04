# Electricity Demand Forecasting Project
**Hackathon 2025 with Alpiq**

## Project Overview

## 📊 Project Overview

This project focuses on predicting **hourly electricity demand** for thousands of consumers in Italy and Spain.

During the 2025 Alpiq Datathon, we were tasked with developing the best possible model to forecast **an entire month of hourly demand** for each consumer. We started by implementing a simple baseline model that used the average consumption for each day of the week and hour, based on the previous month's data.

From there, we iteratively improved our approach, exploring more advanced modeling techniques and feature engineering to boost prediction accuracy and reduce test set loss.


## How-to

### 📦 Prepare the Dataset

1. Place the dataset files in the **root directory** of the repository.
2. Unzip the provided **model_weights.zip** file in the root – the preprocessing step uses these weights for data imputation.

---

### 🚀 Run the Main Notebook

Open and run the notebook:

📓 [`per_consumer_modeling.ipynb`](./notebooks/per_consumer_modeling.ipynb)

This notebook walks through:

- Loading the data
- Preprocessing (including imputation with the General Model)
- Feature analysis
- Building and evaluating per-consumer models

> ✅ This is the only notebook you need to run to use the full pipeline.

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


## Acknowledgments

Thanks to **Alpiq** for the interesting challenge and the opportunity to tackle such a complex forecasting problem!
