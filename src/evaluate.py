import numpy as np
import pandas as pd


def evaluate(
    pred_it: pd.DataFrame,
    pred_es: pd.DataFrame,
    true_it: pd.DataFrame,
    true_es: pd.DataFrame,
    top_k: int = 3,  # Show top-k worst consumers
) -> pd.DataFrame:
    """Evaluate forecast, print scores and show diagnostics."""

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
