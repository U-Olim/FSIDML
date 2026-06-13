"""Validation helpers for replication-level and aggregated simulation outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

REPLICATION_METRIC_COLUMNS = [
    "theta_hat",
    "se",
    "ci_lower",
    "ci_upper",
    "theta_true",
    "covered",
]
SCENARIO_SUMMARY_COLUMNS = [
    "scenario_id",
    "scenario_name",
    "n",
    "p",
    "n_obs",
    "n_covariates",
    "n_folds",
    "dgp_name",
    "learner_name",
    "n_rep",
    "bias",
    "median_bias",
    "mae",
    "rmse",
    "coverage",
    "ci_length",
    "mean_ci_length",
    "empirical_sd",
    "mean_se",
    "se_ratio",
    "variance_ratio",
    "non_convergence_rate",
    "n_replications_total",
    "n_replications_success",
    "n_replications_failed",
    "t_stat_mean",
    "t_stat_sd",
    "mean_fold_train_size",
    "mean_fold_test_size",
    "mean_fold_ratio",
    "max_fold_ratio",
    "mean_condition_number",
    "max_condition_number",
    "mean_min_eigenvalue",
    "min_min_eigenvalue",
    "rank_deficiency_rate",
    "mean_nuisance_mse_y",
    "mean_nuisance_mse_d",
    "mean_nuisance_r2_y",
    "mean_nuisance_r2_d",
]
MAIN_RESULTS_COLUMNS = SCENARIO_SUMMARY_COLUMNS.copy()


def validate_replication_invariants(df: pd.DataFrame) -> None:
    """Raise ValueError if core replication-level inference invariants fail."""

    required_columns = REPLICATION_METRIC_COLUMNS
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    finite_columns = ["theta_hat", "se", "ci_lower", "ci_upper"]
    if not np.isfinite(df.loc[:, finite_columns].to_numpy()).all():
        raise ValueError("theta_hat/se/ci bounds must all be finite")

    if not (df["se"] > 0).all():
        raise ValueError("se must be strictly positive")

    if not (df["ci_upper"] > df["ci_lower"]).all():
        raise ValueError("Each replication must satisfy ci_upper > ci_lower")

    expected_coverage = (
        (df["ci_lower"] <= df["theta_true"]) & (df["theta_true"] <= df["ci_upper"])
    ).astype(int)
    if not (df["covered"].astype(int) == expected_coverage).all():
        raise ValueError("covered indicator is inconsistent with reported confidence bounds")


def validate_aggregate_invariants(raw_df: pd.DataFrame, aggregated_df: pd.DataFrame) -> None:
    """Raise ValueError if aggregated metrics are inconsistent with raw replications."""

    required_columns = {
        "scenario_id",
        "coverage",
        "mean_ci_length",
        "variance_ratio",
        "t_stat_mean",
        "t_stat_sd",
    }
    if not required_columns.issubset(aggregated_df.columns):
        raise ValueError(
            f"Aggregated metrics missing required columns: {sorted(required_columns)}"
        )
    for scenario_id, group in raw_df.groupby("scenario_id", sort=True):
        matched = aggregated_df.loc[aggregated_df["scenario_id"] == scenario_id]
        if len(matched) != 1:
            raise ValueError(f"Expected one aggregate row for scenario_id={scenario_id}")

        row = matched.iloc[0]
        expected_coverage = float(
            (
                (group["ci_lower"] <= group["theta_true"])
                & (group["theta_true"] <= group["ci_upper"])
            ).mean()
        )
        expected_ci_length = float((group["ci_upper"] - group["ci_lower"]).mean())
        mean_se = float(group["se"].mean())
        empirical_sd = float(group["theta_hat"].std(ddof=1))
        expected_variance_ratio = empirical_sd / mean_se if mean_se > 0 else np.nan
        t_stat = (group["theta_hat"] - group["theta_true"]) / group["se"]
        expected_t_stat_mean = float(t_stat.mean())
        expected_t_stat_sd = float(t_stat.std(ddof=1))

        if not np.isclose(float(row["coverage"]), expected_coverage):
            raise ValueError(f"Coverage mismatch for scenario_id={scenario_id}")
        if not np.isclose(float(row["mean_ci_length"]), expected_ci_length):
            raise ValueError(f"Mean CI length mismatch for scenario_id={scenario_id}")
        if np.isnan(expected_variance_ratio):
            if not np.isnan(float(row["variance_ratio"])):
                raise ValueError(f"Variance ratio mismatch for scenario_id={scenario_id}")
        elif not np.isclose(float(row["variance_ratio"]), expected_variance_ratio):
            raise ValueError(f"Variance ratio mismatch for scenario_id={scenario_id}")
        if not np.isclose(float(row["t_stat_mean"]), expected_t_stat_mean):
            raise ValueError(f"t_stat_mean mismatch for scenario_id={scenario_id}")
        if not np.isclose(float(row["t_stat_sd"]), expected_t_stat_sd):
            raise ValueError(f"t_stat_sd mismatch for scenario_id={scenario_id}")


def validate_replication_structure(df: pd.DataFrame, expected_n_rep: int) -> None:
    """Validate replication-level row structure and uniqueness constraints."""

    if expected_n_rep <= 0:
        raise ValueError("expected_n_rep must be > 0")

    required_columns = {"scenario_id", "replication"}
    if not required_columns.issubset(df.columns):
        raise ValueError("Missing scenario_id or replication columns")

    if df.duplicated(subset=["scenario_id", "replication"]).any():
        raise ValueError("Found duplicated (scenario_id, replication) pairs")

    counts = df.groupby("scenario_id").size()
    bad_counts = counts[counts != expected_n_rep]
    if not bad_counts.empty:
        raise ValueError(
            "Each scenario_id must have exactly expected_n_rep rows. "
            f"Violations: {bad_counts.to_dict()}"
        )


def validate_summary_structure(
    summary_df: pd.DataFrame, expected_scenarios: int, design_key_columns: list[str]
) -> None:
    """Validate aggregated scenario summary shape and uniqueness constraints."""

    if len(summary_df) != expected_scenarios:
        raise ValueError(
            f"Summary row count {len(summary_df)} does not match "
            f"expected scenarios {expected_scenarios}"
        )
    if summary_df["scenario_id"].duplicated().any():
        raise ValueError("Duplicate scenario_id found in scenario summary")
    if summary_df.duplicated(subset=design_key_columns).any():
        raise ValueError("Duplicate design-key rows found in scenario summary")


def validate_scenario_summary_schema(summary_df: pd.DataFrame) -> None:
    """Ensure scenario-summary output schema is as expected."""

    if list(summary_df.columns) != SCENARIO_SUMMARY_COLUMNS:
        raise ValueError(
            "scenario_summary has unexpected columns. "
            f"Expected {SCENARIO_SUMMARY_COLUMNS}, got {list(summary_df.columns)}"
        )
    required_metrics = {
        "bias",
        "median_bias",
        "mae",
        "rmse",
        "coverage",
        "ci_length",
        "empirical_sd",
        "mean_se",
        "se_ratio",
        "non_convergence_rate",
    }
    missing = sorted(required_metrics.difference(summary_df.columns))
    if missing:
        raise ValueError(f"scenario_summary missing required metrics columns: {missing}")


def validate_main_results_schema(main_results_df: pd.DataFrame) -> None:
    """Ensure main-results output schema is as expected."""

    if list(main_results_df.columns) != MAIN_RESULTS_COLUMNS:
        raise ValueError(
            "main_results has unexpected columns. "
            f"Expected {MAIN_RESULTS_COLUMNS}, got {list(main_results_df.columns)}"
        )
    required = {
        "scenario_id",
        "scenario_name",
        "n",
        "p",
        "n_folds",
        "dgp_name",
        "learner_name",
        "n_rep",
    }
    if not required.issubset(main_results_df.columns):
        raise ValueError(f"main_results missing required columns: {sorted(required)}")


def validate_summary_and_results_alignment(
    summary_df: pd.DataFrame, main_results_df: pd.DataFrame
) -> None:
    """Ensure summary and results are aligned and non-duplicative."""

    validate_scenario_summary_schema(summary_df)
    validate_main_results_schema(main_results_df)
    if set(summary_df["scenario_id"]) != set(main_results_df["scenario_id"]):
        raise ValueError("scenario_summary and main_results scenario_id sets do not match")
