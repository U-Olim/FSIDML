"""Tests for result diagnostic ranking reports."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from dml_project.reporting.result_diagnostics import create_result_diagnostics

TEST_TMP_ROOT = Path(".test_tmp") / "result_diagnostics"

EXPECTED_REPORTS = {
    "learner_ranking_smoke",
    "dgp_ranking_smoke",
    "k_ranking_smoke",
    "fold_ratio_bins_smoke",
    "worst_20_scenarios_smoke",
    "best_20_scenarios_smoke",
    "rf_diagnostic_smoke",
    "coverage_gap_smoke",
}


def _fresh_test_dir() -> Path:
    shutil.rmtree(TEST_TMP_ROOT, ignore_errors=True)
    TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    return TEST_TMP_ROOT


def _example_summary() -> pd.DataFrame:
    rows = [
        {
            "scenario_name": "good_lasso",
            "dgp_name": "dense_linear_independent",
            "learner_name": "lasso",
            "n_obs": 100,
            "n_covariates": 20,
            "n_folds": 2,
            "mean_fold_ratio": 0.20,
            "bias": 0.01,
            "mae": 0.05,
            "rmse": 0.06,
            "coverage": 0.94,
            "ci_length": 0.30,
            "se_ratio": 1.00,
            "non_convergence_rate": 0.0,
            "mean_condition_number": 10.0,
            "rank_deficiency_rate": 0.0,
            "mean_nuisance_r2_y": 0.70,
            "mean_nuisance_r2_d": 0.60,
        },
        {
            "scenario_name": "bad_ols",
            "dgp_name": "sparse_linear_independent",
            "learner_name": "ols",
            "n_obs": 100,
            "n_covariates": 40,
            "n_folds": 2,
            "mean_fold_ratio": 0.40,
            "bias": 0.30,
            "mae": 0.80,
            "rmse": 1.20,
            "coverage": 0.40,
            "ci_length": 0.20,
            "se_ratio": 0.50,
            "non_convergence_rate": 0.1,
            "mean_condition_number": 100.0,
            "rank_deficiency_rate": 0.2,
            "mean_nuisance_r2_y": 0.10,
            "mean_nuisance_r2_d": 0.20,
        },
        {
            "scenario_name": "mid_rf",
            "dgp_name": "sparse_linear_correlated",
            "learner_name": "random_forest",
            "n_obs": 100,
            "n_covariates": 80,
            "n_folds": 5,
            "mean_fold_ratio": 0.80,
            "bias": -0.10,
            "mae": 0.30,
            "rmse": 0.50,
            "coverage": 0.70,
            "ci_length": 0.40,
            "se_ratio": 0.70,
            "non_convergence_rate": 0.0,
            "mean_condition_number": 50.0,
            "rank_deficiency_rate": 0.0,
            "mean_nuisance_r2_y": 0.30,
            "mean_nuisance_r2_d": 0.25,
        },
        {
            "scenario_name": "high_rho_rf",
            "dgp_name": "weak_signal_sparse",
            "learner_name": "random_forest",
            "n_obs": 100,
            "n_covariates": 120,
            "n_folds": 10,
            "mean_fold_ratio": 1.20,
            "bias": 0.05,
            "mae": 0.40,
            "rmse": 0.60,
            "coverage": 1.00,
            "ci_length": 0.60,
            "se_ratio": 1.50,
            "non_convergence_rate": 0.0,
            "mean_condition_number": 70.0,
            "rank_deficiency_rate": 0.0,
            "mean_nuisance_r2_y": 0.35,
            "mean_nuisance_r2_d": 0.30,
        },
    ]
    return pd.DataFrame(rows)


def _run_diagnostics() -> dict[str, pd.DataFrame]:
    temp_dir = _fresh_test_dir()
    input_path = temp_dir / "scenario_summary_smoke.csv"
    output_dir = temp_dir / "outputs"
    _example_summary().to_csv(input_path, index=False)
    return create_result_diagnostics(input_path, output_dir)


def test_create_result_diagnostics_writes_all_csvs() -> None:
    temp_dir = _fresh_test_dir()
    try:
        input_path = temp_dir / "scenario_summary_smoke.csv"
        output_dir = temp_dir / "outputs"
        _example_summary().to_csv(input_path, index=False)

        tables = create_result_diagnostics(input_path, output_dir)

        assert set(tables) == EXPECTED_REPORTS
        for report_name in EXPECTED_REPORTS:
            assert (output_dir / f"{report_name}.csv").exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_learner_ranking_sorts_by_mae_then_coverage_gap() -> None:
    temp_dir = TEST_TMP_ROOT
    try:
        tables = _run_diagnostics()

        ranking = tables["learner_ranking_smoke"]

        assert list(ranking["learner_name"]) == ["lasso", "random_forest", "ols"]
        assert ranking.iloc[0]["mean_mae"] == 0.05
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_worst_20_selects_low_coverage_high_error_scenarios() -> None:
    temp_dir = TEST_TMP_ROOT
    try:
        tables = _run_diagnostics()

        worst = tables["worst_20_scenarios_smoke"]

        assert worst.iloc[0]["scenario_name"] == "bad_ols"
        assert worst.iloc[0]["coverage"] == 0.40
        assert worst.iloc[0]["mae"] == 0.80
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_best_20_selects_low_mae_scenarios() -> None:
    temp_dir = TEST_TMP_ROOT
    try:
        tables = _run_diagnostics()

        best = tables["best_20_scenarios_smoke"]

        assert best.iloc[0]["scenario_name"] == "good_lasso"
        assert best.iloc[0]["mae"] == 0.05
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_fold_ratio_bins_are_correct() -> None:
    temp_dir = TEST_TMP_ROOT
    try:
        tables = _run_diagnostics()

        bins = tables["fold_ratio_bins_smoke"].set_index("fold_ratio_bin")

        assert bins.loc["rho <= 0.25", "n_scenarios"] == 1
        assert bins.loc["0.25 < rho <= 0.5", "n_scenarios"] == 1
        assert bins.loc["0.5 < rho <= 1.0", "n_scenarios"] == 1
        assert bins.loc["rho > 1.0", "n_scenarios"] == 1
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_rf_diagnostic_only_contains_rf_rows() -> None:
    temp_dir = TEST_TMP_ROOT
    try:
        tables = _run_diagnostics()

        rf = tables["rf_diagnostic_smoke"]

        assert len(rf) == 2
        assert set(rf["dgp_name"]) == {"sparse_linear_correlated", "weak_signal_sparse"}
        assert np.isclose(rf["mean_mae"].mean(), 0.35)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_coverage_gap_is_computed_and_sorted() -> None:
    temp_dir = TEST_TMP_ROOT
    try:
        tables = _run_diagnostics()

        gaps = tables["coverage_gap_smoke"]

        assert gaps.iloc[0]["scenario_name"] == "bad_ols"
        assert np.isclose(gaps.iloc[0]["coverage_gap"], -0.55)
        assert np.isclose(gaps.iloc[0]["absolute_coverage_gap"], 0.55)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_missing_required_columns_raise_clear_error() -> None:
    temp_dir = _fresh_test_dir()
    try:
        input_path = temp_dir / "scenario_summary_smoke.csv"
        output_dir = temp_dir / "outputs"
        _example_summary().drop(columns=["coverage"]).to_csv(input_path, index=False)

        try:
            create_result_diagnostics(input_path, output_dir)
        except ValueError as error:
            assert "coverage" in str(error)
        else:
            raise AssertionError("Expected ValueError for missing coverage")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
