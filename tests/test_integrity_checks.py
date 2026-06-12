"""Tests for hard data-integrity checks used by pipeline tasks."""

from __future__ import annotations

import pandas as pd
import pytest

from dml_project.utils.checks import (
    validate_main_results_schema,
    validate_replication_structure,
    validate_scenario_summary_schema,
    validate_summary_and_results_alignment,
    validate_summary_structure,
)


def test_validate_replication_structure_passes_for_balanced_unique_rows() -> None:
    df = pd.DataFrame(
        [
            {"scenario_id": 0, "replication": 0},
            {"scenario_id": 0, "replication": 1},
            {"scenario_id": 1, "replication": 0},
            {"scenario_id": 1, "replication": 1},
        ]
    )
    validate_replication_structure(df, expected_n_rep=2)


def test_validate_replication_structure_raises_on_duplicate_pairs() -> None:
    df = pd.DataFrame(
        [
            {"scenario_id": 0, "replication": 0},
            {"scenario_id": 0, "replication": 0},
        ]
    )
    with pytest.raises(ValueError):
        validate_replication_structure(df, expected_n_rep=1)


def test_validate_replication_structure_raises_on_bad_counts() -> None:
    df = pd.DataFrame(
        [
            {"scenario_id": 0, "replication": 0},
            {"scenario_id": 1, "replication": 0},
            {"scenario_id": 1, "replication": 1},
        ]
    )
    with pytest.raises(ValueError):
        validate_replication_structure(df, expected_n_rep=2)


def test_validate_summary_structure_raises_on_duplicate_design_keys() -> None:
    summary = pd.DataFrame(
        [
            {
                "scenario_id": 0,
                "dgp_name": "dense_linear_independent",
                "learner_name": "lasso",
                "matched_specification": True,
                "n": 150,
                "p": 150,
                "theta_true": 0.0,
                "n_rep": 500,
            },
            {
                "scenario_id": 1,
                "dgp_name": "dense_linear_independent",
                "learner_name": "lasso",
                "matched_specification": True,
                "n": 150,
                "p": 150,
                "theta_true": 0.0,
                "n_rep": 500,
            },
        ]
    )
    with pytest.raises(ValueError):
        validate_summary_structure(
            summary_df=summary,
            expected_scenarios=2,
            design_key_columns=[
                "dgp_name",
                "learner_name",
                "matched_specification",
                "n",
                "p",
                "theta_true",
                "n_rep",
            ],
        )


def test_validate_scenario_summary_schema_requires_metrics_columns() -> None:
    summary = pd.DataFrame(
        [
            {
                "scenario_id": 0,
                "dgp_name": "dense_linear_independent",
                "learner_name": "lasso",
                "matched_specification": True,
                "n": 150,
                "p": 150,
                "theta_true": 0.0,
                "n_rep": 500,
            }
        ]
    )
    with pytest.raises(ValueError):
        validate_scenario_summary_schema(summary)


def test_validate_main_results_schema_requires_dgp_and_learner() -> None:
    results = pd.DataFrame(
        [
            {
                "scenario_id": 0,
                "bias": 0.1,
                "rmse": 0.2,
                "empirical_sd": 0.3,
                "mean_se": 0.3,
                "variance_ratio": 1.0,
                "t_stat_mean": 0.0,
                "t_stat_sd": 1.0,
                "coverage": 0.95,
                "mean_ci_length": 0.7,
            }
        ]
    )
    with pytest.raises(ValueError):
        validate_main_results_schema(results)


def test_validate_summary_and_results_alignment_requires_same_scenarios() -> None:
    summary = pd.DataFrame(
        [
            {
                "scenario_id": 0,
                "dgp_name": "dense_linear_independent",
                "learner_name": "lasso",
                "matched_specification": True,
                "n": 150,
                "p": 150,
                "theta_true": 0.0,
                "n_rep": 500,
                "bias": 0.1,
                "rmse": 0.2,
                "empirical_sd": 0.3,
                "mean_se": 0.3,
                "coverage": 0.95,
                "mean_ci_length": 0.7,
                "variance_ratio": 1.0,
                "t_stat_mean": 0.0,
                "t_stat_sd": 1.0,
            }
        ]
    )
    results = pd.DataFrame(
        [
            {
                "scenario_id": 1,
                "dgp_name": "dense_linear_independent",
                "learner_name": "lasso",
                "matched_specification": True,
                "bias": 0.1,
                "rmse": 0.2,
                "empirical_sd": 0.3,
                "mean_se": 0.3,
                "coverage": 0.95,
                "mean_ci_length": 0.7,
            }
        ]
    )
    with pytest.raises(ValueError):
        validate_summary_and_results_alignment(summary_df=summary, main_results_df=results)
