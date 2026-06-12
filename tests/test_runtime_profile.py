"""Tests for runtime profiling reports."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from dml_project.reporting.runtime_profile import create_runtime_profile

TEST_TMP_ROOT = Path(".test_tmp") / "runtime_profile"


def _fresh_test_dir(name: str) -> Path:
    path = TEST_TMP_ROOT / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_csv(temp_dir: Path, df: pd.DataFrame) -> str:
    path = temp_dir / "simulations_smoke.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_create_runtime_profile_with_runtime_column() -> None:
    """Runtime profiles should aggregate by learner and K."""

    temp_dir = _fresh_test_dir("with_runtime_column")
    try:
        df = pd.DataFrame(
            [
                {
                    "scenario_id": 1,
                    "scenario_name": "s1",
                    "replication": 0,
                    "learner_name": "ols",
                    "dgp_name": "dense_linear_independent",
                    "n_obs": 100,
                    "n_covariates": 20,
                    "n_folds": 2,
                    "duration_seconds": 1.0,
                },
                {
                    "scenario_id": 1,
                    "scenario_name": "s1",
                    "replication": 1,
                    "learner_name": "ols",
                    "dgp_name": "dense_linear_independent",
                    "n_obs": 100,
                    "n_covariates": 20,
                    "n_folds": 2,
                    "duration_seconds": 3.0,
                },
                {
                    "scenario_id": 2,
                    "scenario_name": "s2",
                    "replication": 0,
                    "learner_name": "lasso",
                    "dgp_name": "sparse_linear_independent",
                    "n_obs": 100,
                    "n_covariates": 20,
                    "n_folds": 5,
                    "duration_seconds": 6.0,
                },
            ]
        )

        profile = create_runtime_profile(_write_csv(temp_dir, df))

        learner_table = profile["runtime_by_learner"]
        ols_row = learner_table.loc[learner_table["learner_name"] == "ols"].iloc[0]
        assert ols_row["total_runtime_seconds"] == 4.0
        assert ols_row["mean_runtime_seconds"] == 2.0
        assert ols_row["median_runtime_seconds"] == 2.0

        k_table = profile["runtime_by_k"]
        assert set(k_table["n_folds"]) == {2, 5}
        assert (
            k_table.loc[k_table["n_folds"] == 5, "total_runtime_seconds"].iloc[0]
            == 6.0
        )

        slowest = profile["slowest_scenarios"]
        assert slowest.iloc[0]["scenario_id"] == 2
        assert slowest.iloc[0]["total_runtime_seconds"] == 6.0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_create_runtime_profile_without_runtime_column() -> None:
    """Missing runtime should fall back to count and completion tables."""

    temp_dir = _fresh_test_dir("without_runtime_column")
    try:
        df = pd.DataFrame(
            [
                {
                    "scenario_id": 1,
                    "scenario_name": "s1",
                    "replication": replication,
                    "learner_name": "ols",
                    "dgp_name": "dense_linear_independent",
                    "n_obs": 100,
                    "n_covariates": 20,
                    "n_folds": 2,
                }
                for replication in range(10)
            ]
        )

        profile = create_runtime_profile(_write_csv(temp_dir, df))

        assert "runtime_by_learner" not in profile
        assert "rows_by_learner" in profile
        assert "rows_by_dgp" in profile
        assert "rows_by_k" in profile
        assert "rows_by_n_p" in profile
        assert "rows_by_learner_and_k" in profile
        assert "scenario_completion_summary" in profile
        assert profile["rows_by_learner"].loc[0, "n_rows"] == 10
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_create_runtime_profile_detects_incomplete_scenarios() -> None:
    """Completion summary should count scenarios below smoke replication count."""

    temp_dir = _fresh_test_dir("incomplete_scenarios")
    try:
        rows = []
        for scenario_id, n_replications in [(1, 10), (2, 9)]:
            for replication in range(n_replications):
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "scenario_name": f"s{scenario_id}",
                        "replication": replication,
                        "learner_name": "ols",
                        "dgp_name": "dense_linear_independent",
                        "n_obs": 100,
                        "n_covariates": 20,
                        "n_folds": 2,
                    }
                )
        df = pd.DataFrame(rows)

        profile = create_runtime_profile(_write_csv(temp_dir, df))
        summary = profile["scenario_completion_summary"].iloc[0]

        assert summary["unique_scenarios"] == 2
        assert summary["min_replications_per_scenario"] == 9
        assert np.isclose(summary["median_replications_per_scenario"], 9.5)
        assert summary["max_replications_per_scenario"] == 10
        assert summary["incomplete_scenarios_count"] == 1
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
