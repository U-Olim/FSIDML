"""Tests for revised publication table generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tasks.task_tables import (
    table_diagnostics,
    table_main_results,
    table_simulation_design,
)


def _aggregated_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario_name": "quadratic_n100_p20_k5_lasso",
                "n_obs": 100,
                "n_covariates": 20,
                "n_folds": 5,
                "dgp_name": "quadratic_confounding",
                "learner_name": "lasso",
                "mean_fold_train_size": 80.0,
                "mean_fold_ratio": 0.25,
                "bias": -0.12345,
                "median_bias": -0.11111,
                "mae": 0.22222,
                "rmse": 0.33333,
                "coverage": 0.94444,
                "ci_length": 0.55555,
                "se_ratio": 1.23456,
                "non_convergence_rate": 0.01234,
                "mean_condition_number": 123.456,
                "max_condition_number": 456.789,
                "mean_min_eigenvalue": 0.001234,
                "rank_deficiency_rate": 0.1234,
                "mean_nuisance_mse_y": 1.23456,
                "mean_nuisance_mse_d": 2.34567,
                "mean_nuisance_r2_y": 0.34567,
                "mean_nuisance_r2_d": 0.45678,
            },
            {
                "scenario_name": "linear_n100_p20_k2_rf",
                "n_obs": 100,
                "n_covariates": 20,
                "n_folds": 2,
                "dgp_name": "linear_confounding",
                "learner_name": "random_forest",
                "bias": 0.1,
                "median_bias": 0.2,
                "mae": 0.3,
                "rmse": 0.4,
                "coverage": 0.9,
                "ci_length": 0.7,
                "se_ratio": 0.8,
                "non_convergence_rate": 0.0,
                "mean_condition_number": 10.0,
                "max_condition_number": 20.0,
                "mean_min_eigenvalue": 0.02,
                "rank_deficiency_rate": 0.0,
                "mean_nuisance_mse_y": 0.5,
                "mean_nuisance_mse_d": 0.6,
                "mean_nuisance_r2_y": 0.7,
                "mean_nuisance_r2_d": 0.8,
            },
        ]
    )


def test_table_simulation_design_columns_and_design_values() -> None:
    """Design table should compute n_train and rho_fold."""

    table = table_simulation_design(_aggregated_results())

    assert list(table.columns) == ["DGP", "Learner", "n", "p", "K", "n_train", "rho_fold"]
    linear_row = table.loc[table["DGP"] == "Linear"].iloc[0]
    assert linear_row["n_train"] == 50.0
    assert linear_row["rho_fold"] == 0.4
    quadratic_row = table.loc[table["DGP"] == "Quadratic"].iloc[0]
    assert quadratic_row["n_train"] == 80.0
    assert quadratic_row["rho_fold"] == 0.25


def test_table_labels_are_readable() -> None:
    """DGP and learner names should be publication labels."""

    table = table_simulation_design(_aggregated_results())

    assert set(table["DGP"]) == {"Linear", "Quadratic"}
    assert set(table["Learner"]) == {"Random Forest", "Lasso"}


def test_table_main_results_contains_revised_metrics() -> None:
    """Main table should expose the Step 6 metric set."""

    table = table_main_results(_aggregated_results())

    assert {
        "Bias",
        "Median Bias",
        "MAE",
        "RMSE",
        "Coverage",
        "CI Length",
        "SE Ratio",
        "Non-convergence",
    }.issubset(table.columns)
    quadratic_row = table.loc[table["DGP"] == "Quadratic"].iloc[0]
    assert quadratic_row["Bias"] == -0.123
    assert quadratic_row["SE Ratio"] == 1.235


def test_table_diagnostics_contains_required_columns() -> None:
    """Diagnostics table should include instability and nuisance metrics."""

    table = table_diagnostics(_aggregated_results())

    assert {
        "Mean Cond. No.",
        "Rank Def. Rate",
        "Nuisance MSE Y",
        "Nuisance R2 D",
    }.issubset(table.columns)
    quadratic_row = table.loc[table["DGP"] == "Quadratic"].iloc[0]
    assert quadratic_row["Mean Cond. No."] == 123.46
    assert quadratic_row["Mean Min Eigenvalue"] == 0.0012


def test_tables_preserve_k_scenarios() -> None:
    """K=2 and K=5 rows should remain separate."""

    table = table_main_results(_aggregated_results())

    assert set(table["K"]) == {2, 5}
    assert len(table) == 2


def test_missing_optional_diagnostics_produce_nan() -> None:
    """Missing diagnostic inputs should produce NaN output columns."""

    df = _aggregated_results().drop(columns=["mean_condition_number", "mean_nuisance_r2_d"])
    table = table_diagnostics(df)

    assert np.isnan(table["Mean Cond. No."]).all()
    assert np.isnan(table["Nuisance R2 D"]).all()


def test_table_sorting_is_deterministic() -> None:
    """Rows should sort by DGP, learner, n, p, and K regardless of input order."""

    df = _aggregated_results().iloc[::-1].reset_index(drop=True)
    table = table_simulation_design(df)

    assert list(table["DGP"]) == ["Linear", "Quadratic"]
