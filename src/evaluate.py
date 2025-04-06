import numpy as np
import pandas as pd


def evaluate(
    pred_it: pd.DataFrame,
    pred_es: pd.DataFrame,
    true_it: pd.DataFrame,
    true_es: pd.DataFrame,
    top_k: int = 3,  # Show top-k worst consumers
) -> pd.DataFrame:
    """
    Evaluate forecast accuracy and provide diagnostic information.

    This function computes the absolute and portfolio errors for forecasted
    energy consumption data for two countries (Italy and Spain). It also
    generates a weighted scoring table and identifies the top-k consumers
    with the highest errors for diagnostic purposes.

    Args:
        pred_it (pd.DataFrame): Forecasted energy consumption for Italy.
            The DataFrame should have the same structure (columns and index)
            as `true_it`.
        pred_es (pd.DataFrame): Forecasted energy consumption for Spain.
            The DataFrame should have the same structure (columns and index)
            as `true_es`.
        true_it (pd.DataFrame): Actual energy consumption for Italy.
            The DataFrame should have the same structure (columns and index)
            as `pred_it`.
        true_es (pd.DataFrame): Actual energy consumption for Spain.
            The DataFrame should have the same structure (columns and index)
            as `pred_es`.
        top_k (int, optional): The number of top consumers with the highest
            errors to display in the diagnostic report. Defaults to 3.

    Returns:
        pd.DataFrame: A scoring table summarizing the absolute and portfolio
        errors for both countries, along with their weighted scores.

    Raises:
        AssertionError: If the structure (columns or index) of the forecasted
            data does not match the actual data for either country.
        AssertionError: If there are NaN values in the forecasted data.

    Notes:
        - The scoring formula is as follows:
          `1.0 * Absolute Error (IT) + 5.0 * Absolute Error (ES) +
           10.0 * Portfolio Error (IT) + 50.0 * Portfolio Error (ES)`
        - The function prints a detailed evaluation report, including:
          - Absolute and portfolio errors for each country.
          - Weighted scores for Italy and Spain.
          - Top-k consumers with the highest errors for each country.
          - The total forecast score.

    Example:
        >>> evaluate(pred_it, pred_es, true_it, true_es, top_k=5)
        This will compute the evaluation metrics and print a detailed
        diagnostic report for the forecasted data.
    """

    absolute_error = {}
    portfolio_error = {}
    consumer_errors = {}  # For diagnostic plots and tables

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

        # Core errors
        abs_err = (student_solution - testing_set).abs().sum().sum()
        port_err = (student_solution - testing_set).sum(axis=1).abs().sum()

        absolute_error[country] = abs_err
        portfolio_error[country] = port_err

        # Store per-consumer error
        consumer_abs_error = (student_solution - testing_set).abs().sum()
        consumer_errors[country] = consumer_abs_error.sort_values(ascending=False)

    # === Official scoring table ===
    forecast_score = (
        1.0 * absolute_error["IT"]
        + 5.0 * absolute_error["ES"]
        + 10.0 * portfolio_error["IT"]
        + 50.0 * portfolio_error["ES"]
    )

    score_table = pd.DataFrame(
        {"Absolute Error": absolute_error, "Portfolio Error": portfolio_error}
    ).T

    score_table["Weight IT"] = [1.0, 10.0]  # IT weights
    score_table["Weighted IT"] = score_table["IT"] * score_table["Weight IT"]
    score_table["Weight ES"] = [5.0, 50.0]  # ES weights
    score_table["Weighted ES"] = score_table["ES"] * score_table["Weight ES"]
    score_table = score_table[
        ["IT", "Weight IT", "Weighted IT", "ES", "Weight ES", "Weighted ES"]
    ]
    score_table.columns.name = "Metric"

    score_it = score_table[["IT", "Weight IT", "Weighted IT"]]
    score_it = score_it.rename(
        columns={"IT": "Score", "Weight IT": "Weight", "Weighted IT": "Weighted Score"}
    )
    score_es = score_table[["ES", "Weight ES", "Weighted ES"]]
    score_es = score_es.rename(
        columns={"ES": "Score", "Weight ES": "Weight", "Weighted ES": "Weighted Score"}
    )

    print("\n" + "=" * 60)
    print("FORECAST EVALUATION REPORT".center(60))
    print("=" * 60 + "\n")
    print("IT PERFORMANCE")
    print("-" * 60 + "\n")
    print(score_it)
    print(f"\nTop {top_k} consumers with highest error in Italy:\n")
    print(consumer_errors["IT"].head(top_k).round(2).to_string())

    print("\nES PERFORMANCE")
    print("-" * 60 + "\n")
    print(score_es)
    print(f"\nTop {top_k} consumers with highest error in Spain:\n")
    print(consumer_errors["ES"].head(top_k).round(2).to_string())

    print("\n" + "-" * 60)
    print("TOTAL FORECAST SCORE".center(60))
    print(f"{int(round(forecast_score))}".center(60))
    print("-" * 60 + "\n")
