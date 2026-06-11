"""Tests for aggregation and seed utility modules."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

from dml_project.simulation.aggregate import aggregate_results
from dml_project.utils.seeds import (
    SeedBundle,
    make_numpy_rng,
    make_seed_bundle,
    seed_bundle_to_dict,
)


def _base_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario_id": 1,
                "scenario_name": "linear_n100_p5_k2_ols",
                "n_obs": 100,
                "n_covariates": 5,
                "n_folds": 2,
                "dgp_name": "linear_confounding",
                "learner_name": "ols",
                "replication": 0,
                "theta_0": 1.0,
                "theta_hat": 1.2,
                "se": 0.10,
                "ci_lower": 0.9,
                "ci_upper": 1.5,
                "covered": True,
                "failed": False,
                "mean_fold_train_size": 50.0,
                "mean_fold_test_size": 50.0,
                "mean_fold_ratio": 0.10,
                "max_fold_ratio": 0.10,
                "mean_condition_number": 2.0,
                "max_condition_number": 3.0,
                "mean_min_eigenvalue": 0.40,
                "min_min_eigenvalue": 0.30,
                "rank_deficiency_rate": 0.0,
                "mean_nuisance_mse_y": 1.0,
                "mean_nuisance_mse_d": 0.5,
                "mean_nuisance_r2_y": 0.2,
                "mean_nuisance_r2_d": 0.3,
            },
            {
                "scenario_id": 1,
                "scenario_name": "linear_n100_p5_k2_ols",
                "n_obs": 100,
                "n_covariates": 5,
                "n_folds": 2,
                "dgp_name": "linear_confounding",
                "learner_name": "ols",
                "replication": 1,
                "theta_0": 1.0,
                "theta_hat": 0.8,
                "se": 0.20,
                "ci_lower": 0.4,
                "ci_upper": 0.9,
                "covered": False,
                "failed": False,
                "mean_fold_train_size": 50.0,
                "mean_fold_test_size": 50.0,
                "mean_fold_ratio": 0.10,
                "max_fold_ratio": 0.12,
                "mean_condition_number": 4.0,
                "max_condition_number": 5.0,
                "mean_min_eigenvalue": 0.20,
                "min_min_eigenvalue": 0.10,
                "rank_deficiency_rate": 0.5,
                "mean_nuisance_mse_y": 2.0,
                "mean_nuisance_mse_d": 1.5,
                "mean_nuisance_r2_y": 0.4,
                "mean_nuisance_r2_d": 0.5,
            },
            {
                "scenario_id": 1,
                "scenario_name": "linear_n100_p5_k2_ols",
                "n_obs": 100,
                "n_covariates": 5,
                "n_folds": 2,
                "dgp_name": "linear_confounding",
                "learner_name": "ols",
                "replication": 2,
                "theta_0": 1.0,
                "theta_hat": np.nan,
                "se": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "covered": False,
                "failed": True,
                "failure_reason": "ValueError: forced",
            },
        ]
    )


def _single_row_metric(df: pd.DataFrame, metric: str) -> float:
    aggregated = aggregate_results(df)
    assert len(aggregated) == 1
    return float(aggregated.iloc[0][metric])


def test_import_aggregation_modules() -> None:
    """Ensure aggregation modules import without errors."""

    importlib.import_module("dml_project.simulation.aggregate")
    importlib.import_module("dml_project.simulation.metrics")


def test_make_numpy_rng_returns_generator() -> None:
    """RNG helper should return a NumPy Generator."""

    rng = make_numpy_rng(123)

    assert isinstance(rng, np.random.Generator)


def test_seed_bundle_to_dict_roundtrip_fields() -> None:
    """Dictionary conversion should preserve bundle values."""

    bundle = SeedBundle(
        base_seed=123,
        scenario_id=1,
        replication=2,
        data_seed=101,
        split_seed=202,
        learner_g_seed=303,
        learner_m_seed=404,
    )

    assert seed_bundle_to_dict(bundle) == {
        "base_seed": 123,
        "scenario_id": 1,
        "replication": 2,
        "data_seed": 101,
        "split_seed": 202,
        "learner_g_seed": 303,
        "learner_m_seed": 404,
    }


def test_make_seed_bundle_is_stable_for_same_inputs() -> None:
    """Seed bundle should be deterministic for fixed inputs."""

    first = make_seed_bundle(scenario_id=2, replication=7, base_seed=123)
    second = make_seed_bundle(scenario_id=2, replication=7, base_seed=123)

    assert first == second


def test_make_seed_bundle_streams_are_distinct() -> None:
    """Seed streams should be independent and non-consecutive."""

    bundle = make_seed_bundle(scenario_id=2, replication=7, base_seed=123)
    seeds = [
        bundle.data_seed,
        bundle.split_seed,
        bundle.learner_g_seed,
        bundle.learner_m_seed,
    ]

    assert len(set(seeds)) == 4


def test_make_seed_bundle_changes_across_replications() -> None:
    """Different replication ids should generate a different bundle."""

    first = make_seed_bundle(scenario_id=2, replication=7, base_seed=123)
    second = make_seed_bundle(scenario_id=2, replication=8, base_seed=123)

    assert first != second


def test_aggregation_computes_main_metrics_on_successes() -> None:
    """Bias, median bias, MAE, and RMSE should use successful rows only."""

    aggregated = aggregate_results(_base_rows())
    row = aggregated.iloc[0]
    errors = np.array([0.2, -0.2])

    assert np.isclose(row["bias"], np.mean(errors))
    assert np.isclose(row["median_bias"], np.median(errors))
    assert np.isclose(row["mae"], np.mean(np.abs(errors)))
    assert np.isclose(row["rmse"], np.sqrt(np.mean(errors**2)))


def test_failed_rows_excluded_from_metrics_but_counted_in_rate() -> None:
    """Failures should affect non-convergence counts but not performance metrics."""

    aggregated = aggregate_results(_base_rows())
    row = aggregated.iloc[0]

    assert row["n_replications_total"] == 3
    assert row["n_replications_success"] == 2
    assert row["n_replications_failed"] == 1
    assert np.isclose(row["non_convergence_rate"], 1 / 3)
    assert np.isclose(row["mean_se"], 0.15)


def test_aggregation_computes_coverage_and_ci_length() -> None:
    """Coverage and CI length should average successful rows."""

    aggregated = aggregate_results(_base_rows())
    row = aggregated.iloc[0]

    assert np.isclose(row["coverage"], 0.5)
    assert np.isclose(row["ci_length"], np.mean([0.6, 0.5]))
    assert np.isclose(row["mean_ci_length"], row["ci_length"])


def test_se_ratio_uses_mean_se_over_empirical_sd() -> None:
    """se_ratio should equal mean_se divided by empirical SD."""

    aggregated = aggregate_results(_base_rows())
    row = aggregated.iloc[0]
    empirical_sd = np.std([1.2, 0.8], ddof=1)

    assert np.isclose(row["empirical_sd"], empirical_sd)
    assert np.isclose(row["se_ratio"], 0.15 / empirical_sd)
    assert np.isclose(row["variance_ratio"], empirical_sd / 0.15)


def test_se_ratio_is_nan_with_one_successful_replication() -> None:
    """Empirical SD and se_ratio should be NaN with fewer than two successes."""

    df = _base_rows().iloc[[0, 2]].copy()
    aggregated = aggregate_results(df)
    row = aggregated.iloc[0]

    assert np.isnan(row["empirical_sd"])
    assert np.isnan(row["se_ratio"])


def test_all_failed_group_returns_nan_performance_metrics() -> None:
    """All-failed groups should still report failure counts and rate."""

    df = _base_rows().iloc[[2]].copy()
    aggregated = aggregate_results(df)
    row = aggregated.iloc[0]

    assert row["n_replications_total"] == 1
    assert row["n_replications_success"] == 0
    assert row["n_replications_failed"] == 1
    assert row["non_convergence_rate"] == 1.0
    for metric in ["bias", "median_bias", "mae", "rmse", "coverage", "ci_length"]:
        assert np.isnan(row[metric])


def test_diagnostics_are_aggregated_from_successful_rows() -> None:
    """Estimator diagnostics should use means, maxima, and minima as specified."""

    aggregated = aggregate_results(_base_rows())
    row = aggregated.iloc[0]

    assert np.isclose(row["mean_fold_train_size"], 50.0)
    assert np.isclose(row["mean_fold_ratio"], 0.10)
    assert np.isclose(row["max_fold_ratio"], 0.12)
    assert np.isclose(row["mean_condition_number"], 3.0)
    assert np.isclose(row["max_condition_number"], 5.0)
    assert np.isclose(row["mean_min_eigenvalue"], 0.30)
    assert np.isclose(row["min_min_eigenvalue"], 0.10)
    assert np.isclose(row["rank_deficiency_rate"], 0.25)
    assert np.isclose(row["mean_nuisance_mse_y"], 1.5)
    assert np.isclose(row["mean_nuisance_mse_d"], 1.0)
    assert np.isclose(row["mean_nuisance_r2_y"], 0.3)
    assert np.isclose(row["mean_nuisance_r2_d"], 0.4)


def test_missing_diagnostic_columns_produce_nan() -> None:
    """Optional diagnostic columns should not be required."""

    df = _base_rows().drop(columns=["mean_fold_ratio", "mean_nuisance_mse_y"])
    aggregated = aggregate_results(df)
    row = aggregated.iloc[0]

    assert np.isnan(row["mean_fold_ratio"])
    assert np.isnan(row["mean_nuisance_mse_y"])


def test_missing_failed_column_assumes_all_successful() -> None:
    """Old result tables without failed should aggregate as all successful."""

    df = _base_rows().iloc[:2].drop(columns=["failed"])

    assert _single_row_metric(df, "n_replications_success") == 2
    assert _single_row_metric(df, "non_convergence_rate") == 0.0


def test_missing_covered_column_is_computed_from_ci() -> None:
    """Coverage should be derived from CI bounds when covered is absent."""

    df = _base_rows().iloc[:2].drop(columns=["covered"])

    assert _single_row_metric(df, "coverage") == 0.5


def test_grouping_includes_n_folds() -> None:
    """Scenarios that differ only in K should remain separate groups."""

    df_k2 = _base_rows().iloc[[0]].copy()
    df_k5 = df_k2.copy()
    df_k5["scenario_id"] = 2
    df_k5["scenario_name"] = "linear_n100_p5_k5_ols"
    df_k5["n_folds"] = 5
    df = pd.concat([df_k2, df_k5], ignore_index=True)

    aggregated = aggregate_results(df)

    assert len(aggregated) == 2
    assert set(aggregated["n_folds"]) == {2, 5}
