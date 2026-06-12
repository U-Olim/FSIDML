"""End-to-end smoke test for the revised simulation pipeline."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from dml_project import config
from dml_project.simulation.aggregate import aggregate_results
from dml_project.simulation.runner import run_scenario
from dml_project.simulation.scenario import Scenario
from tasks.task_figures import plot_coverage_by_fold_ratio, plot_k_sensitivity
from tasks.task_tables import (
    table_diagnostics,
    table_main_results,
    table_simulation_design,
)


def test_revised_pipeline_end_to_end_smoke() -> None:
    """Run a tiny scenario through runner, aggregation, tables, and figures."""

    scenario = Scenario(
        scenario_id=999,
        dgp_name="dense_linear_independent",
        learner_name="ols",
        n_obs=100,
        n_covariates=10,
        n_folds=2,
        theta=config.THETA_TRUE,
        n_rep=3,
        base_seed=20260611,
    )

    runner_output = run_scenario(scenario)

    assert len(runner_output) == 3
    assert {
        "scenario_name",
        "n_obs",
        "n_covariates",
        "n_folds",
        "dgp_name",
        "learner_name",
        "replication",
        "theta_0",
        "theta_hat",
        "se",
        "ci_lower",
        "ci_upper",
        "covered",
        "failed",
        "failure_reason",
        "mean_fold_ratio",
        "mean_nuisance_mse_y",
        "mean_nuisance_mse_d",
    }.issubset(runner_output.columns)

    aggregated = aggregate_results(runner_output)

    assert len(aggregated) == 1
    assert {
        "bias",
        "median_bias",
        "mae",
        "rmse",
        "coverage",
        "ci_length",
        "se_ratio",
        "non_convergence_rate",
        "n_replications_total",
        "n_replications_success",
        "n_replications_failed",
    }.issubset(aggregated.columns)

    design_table = table_simulation_design(aggregated)
    main_table = table_main_results(aggregated)
    diagnostics_table = table_diagnostics(aggregated)

    assert not design_table.empty
    assert not main_table.empty
    assert not diagnostics_table.empty

    figures = [
        plot_coverage_by_fold_ratio(aggregated),
        plot_k_sensitivity(aggregated, metric="coverage"),
    ]
    try:
        assert all(isinstance(fig, Figure) for fig in figures)
    finally:
        for fig in figures:
            plt.close(fig)
