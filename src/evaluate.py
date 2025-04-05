import numpy as np
import pandas as pd


def evaluate(
    pred_it: pd.DataFrame,
    pred_es: pd.DataFrame,
    true_it: pd.DataFrame,
    true_es: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate forecast and print scores in a nice table."""

    absolute_error = {}
    portfolio_error = {}

    for country in ["ES", "IT"]:
        student_solution = pred_es if country == "ES" else pred_it
        testing_set = true_es if country == "ES" else true_it

        assert np.all(
            student_solution.columns == testing_set.columns
        ), f"Wrong header or header order for {country}"
        assert np.all(
            student_solution.index == testing_set.index
        ), f"Wrong index or index order for {country}"
        assert (
            student_solution.isna().sum().sum() == 0
        ), f"NaN in forecast for {country}"

        abs_err = (student_solution - testing_set).abs().sum().sum()
        port_err = (student_solution - testing_set).sum(axis=1).abs().sum()

        absolute_error[country] = abs_err
        portfolio_error[country] = port_err

    forecast_score = (
        1.0 * absolute_error["IT"]
        + 5.0 * absolute_error["ES"]
        + 10.0 * portfolio_error["IT"]
        + 50.0 * portfolio_error["ES"]
    )

    # Prepare table
    score_table = pd.DataFrame(
        {"Absolute Error": absolute_error, "Portfolio Error": portfolio_error}
    ).T

    score_table["Weight"] = [1.0, 10.0]  # IT weights
    score_table["Weighted IT"] = score_table["IT"] * score_table["Weight"]
    score_table["Weight"] = [5.0, 50.0]  # ES weights
    score_table["Weighted ES"] = score_table["ES"] * score_table["Weight"]

    score_table = score_table[["IT", "Weight", "Weighted IT", "ES", "Weighted ES"]]
    score_table.columns.name = "Metric"

    total_score = forecast_score

    score_table = pd.DataFrame(
        {"Absolute Error": absolute_error, "Portfolio Error": portfolio_error}
    ).T

    score_table["Weight"] = [1.0, 10.0]  # IT weights
    score_table["Weighted IT"] = score_table["IT"] * score_table["Weight"]
    score_table["Weight"] = [5.0, 50.0]  # ES weights
    score_table["Weighted ES"] = score_table["ES"] * score_table["Weight"]

    score_table = score_table[["IT", "Weight", "Weighted IT", "ES", "Weighted ES"]]
    score_table.columns.name = "Metric"

    total_score = forecast_score
    print("\n" + "=" * 60)
    print("FORECAST EVALUATION SCORE TABLE".center(60))
    print("=" * 60 + "\n")
    print(score_table.round(2))
    print("\n" + "-" * 60)
    print(f"TOTAL FORECAST SCORE: {int(round(total_score))}".center(60))
    print("-" * 60 + "\n")
