"""Tests for revised simulation figure generation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure

from tasks.task_figures import (
    plot_condition_number_vs_se_distortion,
    plot_coverage_by_fold_ratio,
    plot_k_sensitivity,
    plot_nonlinear_dgp_learners,
)


def _aggregated_results() -> pd.DataFrame:
    rows: list[dict] = []
    dgp_names = [
        "linear_confounding",
        "quadratic_confounding",
        "interaction_confounding",
        "step_confounding",
    ]
    learner_names = ["ols", "lasso", "random_forest"]
    for dgp_index, dgp_name in enumerate(dgp_names):
        for learner_index, learner_name in enumerate(learner_names):
            for n_folds in [2, 5]:
                rows.append(
                    {
                        "scenario_name": f"{dgp_name}_{learner_name}_k{n_folds}",
                        "n_obs": 100,
                        "n_covariates": 20,
                        "n_folds": n_folds,
                        "dgp_name": dgp_name,
                        "learner_name": learner_name,
                        "coverage": 0.9 + 0.01 * learner_index - 0.02 * dgp_index,
                        "mae": 0.2 + 0.02 * dgp_index + 0.01 * learner_index,
                        "rmse": 0.3 + 0.02 * dgp_index + 0.01 * learner_index,
                        "ci_length": 0.5 + 0.01 * n_folds,
                        "se_ratio": 0.8 + 0.1 * learner_index,
                        "non_convergence_rate": 0.01 * dgp_index,
                        "mean_fold_ratio": 20 / (100 * (1 - 1 / n_folds)),
                        "mean_condition_number": 10 + 10 * dgp_index + learner_index,
                    }
                )
    return pd.DataFrame(rows)


def test_coverage_by_fold_ratio_returns_figure_with_mean_fold_ratio() -> None:
    """Coverage-by-rho plot should return a Figure."""

    fig = plot_coverage_by_fold_ratio(_aggregated_results(), dgp_name="linear_confounding")

    assert isinstance(fig, Figure)
    plt.close(fig)


def test_coverage_by_fold_ratio_computes_rho_when_missing() -> None:
    """Coverage plot should compute rho from n, p, and K when needed."""

    df = _aggregated_results().drop(columns=["mean_fold_ratio"])
    fig = plot_coverage_by_fold_ratio(df, dgp_name="linear_confounding")

    assert isinstance(fig, Figure)
    assert fig.axes[0].get_xlabel() == "Fold-level dimensionality, p / n_train"
    plt.close(fig)


def test_condition_number_vs_se_distortion_uses_log_scale() -> None:
    """Conditioning plot should use a log-scaled x-axis."""

    fig = plot_condition_number_vs_se_distortion(_aggregated_results())

    assert isinstance(fig, Figure)
    assert fig.axes[0].get_xscale() == "log"
    plt.close(fig)


@pytest.mark.parametrize("metric", ["coverage", "mae"])
def test_k_sensitivity_valid_metrics(metric: str) -> None:
    """K sensitivity should support coverage and MAE."""

    fig = plot_k_sensitivity(_aggregated_results(), metric=metric)

    assert isinstance(fig, Figure)
    plt.close(fig)


def test_k_sensitivity_invalid_metric_raises() -> None:
    """K sensitivity should reject unknown metrics."""

    with pytest.raises(ValueError, match="metric must be one of"):
        plot_k_sensitivity(_aggregated_results(), metric="bias")


def test_nonlinear_dgp_learners_returns_figure() -> None:
    """Nonlinear DGP learner plot should return a Figure."""

    fig = plot_nonlinear_dgp_learners(_aggregated_results(), metric="mae")

    assert isinstance(fig, Figure)
    assert fig.axes[0].get_xlabel() == "DGP"
    plt.close(fig)


def test_missing_required_core_columns_raise_clear_value_error() -> None:
    """Missing required columns should raise a clear ValueError."""

    df = _aggregated_results().drop(columns=["coverage"])

    with pytest.raises(ValueError, match="requires columns"):
        plot_coverage_by_fold_ratio(df)


def test_save_path_writes_file(tmp_path) -> None:
    """Plot functions should save a file when save_path is provided."""

    save_path = tmp_path / "coverage.png"
    fig = plot_coverage_by_fold_ratio(
        _aggregated_results(),
        dgp_name="linear_confounding",
        save_path=save_path,
    )

    assert save_path.exists()
    assert save_path.stat().st_size > 0
    plt.close(fig)
