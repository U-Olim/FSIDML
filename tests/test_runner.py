"""Tests for Monte Carlo simulation runner behavior."""

from __future__ import annotations

import numpy as np

from dml_project import config
from dml_project.simulation import runner
from dml_project.simulation.runner import run_scenario, run_single_replication
from dml_project.simulation.scenario import Scenario


def _small_scenario(n_rep: int = 1, n_folds: int = 5) -> Scenario:
    return Scenario(
        scenario_id=101,
        dgp_name="dense_linear_independent",
        learner_name="ols",
        n_obs=50,
        n_covariates=5,
        n_folds=n_folds,
        theta=config.THETA_TRUE,
        n_rep=n_rep,
        base_seed=123,
    )


def test_runner_uses_scenario_specific_n_folds() -> None:
    """Runner should pass scenario.n_folds into the estimator."""

    scenario = _small_scenario(n_folds=5)
    row = run_single_replication(scenario, replication=0)

    assert row["n_folds"] == 5
    assert row["mean_fold_train_size"] == 40.0


def test_successful_runner_output_contains_required_columns() -> None:
    """Successful replication rows should include inference and diagnostics."""

    row = run_single_replication(_small_scenario(), replication=0)

    required_columns = {
        "theta_hat",
        "se",
        "ci_lower",
        "ci_upper",
        "covered",
        "failed",
        "failure_reason",
        "n_folds",
        "mean_fold_ratio",
        "mean_nuisance_mse_y",
        "mean_nuisance_mse_d",
    }

    assert required_columns <= set(row)
    assert row["failed"] is False
    assert row["failure_reason"] == ""


def test_runner_coverage_matches_confidence_interval() -> None:
    """Coverage flag should match whether theta_0 lies inside the CI."""

    row = run_single_replication(_small_scenario(), replication=0)
    expected = row["ci_lower"] <= row["theta_0"] <= row["ci_upper"]

    assert isinstance(row["covered"], bool)
    assert row["covered"] == expected


def test_runner_includes_scenario_metadata() -> None:
    """Replication rows should carry scenario metadata."""

    row = run_single_replication(_small_scenario(), replication=0)

    for column in [
        "scenario_name",
        "n_obs",
        "n_covariates",
        "dgp_name",
        "learner_name",
        "replication",
    ]:
        assert column in row


def test_runner_failure_handling_returns_failed_row(monkeypatch) -> None:
    """Expected estimator failures should not crash the scenario run."""

    class FailingDMLPLR:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def fit(self, y, d, x) -> None:
            raise ValueError("forced estimator failure")

    monkeypatch.setattr(runner, "DMLPLR", FailingDMLPLR)

    row = run_single_replication(_small_scenario(), replication=0)

    assert row["failed"] is True
    assert np.isnan(row["theta_hat"])
    assert row["covered"] is False
    assert "ValueError: forced estimator failure" in row["failure_reason"]
    assert np.isnan(row["mean_fold_ratio"])


def test_run_scenario_returns_one_row_per_replication() -> None:
    """Multiple replications should produce the requested number of rows."""

    results = run_scenario(_small_scenario(n_rep=3))

    assert len(results) == 3
    assert list(results["replication"]) == [0, 1, 2]
