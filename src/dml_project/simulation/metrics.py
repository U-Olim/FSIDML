"""Metric and inference helpers for PLR Monte Carlo simulations."""

from __future__ import annotations

import numpy as np


def plr_score_components(
    y: np.ndarray, d: np.ndarray, g_hat: np.ndarray, m_hat: np.ndarray, theta_hat: float
) -> dict[str, np.ndarray | float]:
    """Return PLR score components used for variance/SE/CI.

    Orthogonal score for PLR ATE:
    psi_i(theta) = (d_i - m_hat_i) * ((y_i - g_hat_i) - theta * (d_i - m_hat_i))

    Code notation:
    - y_res = y - g_hat
    - d_res = d - m_hat
    - psi = d_res * (y_res - theta_hat * d_res)
    - v_hat = mean(d_res ** 2)
    """

    y_res = y - g_hat
    d_res = d - m_hat
    v_hat = float(np.mean(d_res**2))
    psi = d_res * (y_res - theta_hat * d_res)
    return {"y_res": y_res, "d_res": d_res, "psi": psi, "v_hat": v_hat}


def bias_and_rmse(theta_hat: np.ndarray, theta_true: np.ndarray) -> tuple[float, float]:
    """Compute Monte Carlo bias and RMSE.

    Args:
        theta_hat: Replication-level effect estimates.
        theta_true: Replication-level true effect values.

    Returns:
        Tuple ``(bias, rmse)``.
    """

    errors = theta_hat - theta_true
    bias = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors**2)))
    return bias, rmse


def mc_se_bias(theta_hat: np.ndarray, theta_true: np.ndarray) -> float:
    """Compute Monte Carlo standard error for estimated bias.

    Args:
        theta_hat: Replication-level effect estimates.
        theta_true: Replication-level true effect values.

    Returns:
        Monte Carlo standard error of ``mean(theta_hat - theta_true)``.
    """

    errors = theta_hat - theta_true
    return float(np.std(errors, ddof=1) / np.sqrt(errors.shape[0]))


def mc_se_rmse(theta_hat: np.ndarray, theta_true: np.ndarray) -> float:
    """Compute delta-method Monte Carlo standard error of RMSE.

    Args:
        theta_hat: Replication-level effect estimates.
        theta_true: Replication-level true effect values.

    Returns:
        Monte Carlo standard error for RMSE.
    """

    squared_errors = (theta_hat - theta_true) ** 2
    mean_squared_error = float(np.mean(squared_errors))
    rmse = float(np.sqrt(mean_squared_error))
    if np.isclose(rmse, 0.0):
        return 0.0

    se_mse = float(np.std(squared_errors, ddof=1) / np.sqrt(squared_errors.shape[0]))
    return float(se_mse / (2.0 * rmse))


def mc_se_coverage(covered: np.ndarray) -> float:
    """Compute binomial Monte Carlo standard error for empirical coverage.

    Args:
        covered: Indicator array (0/1) for CI coverage by replication.

    Returns:
        Standard error of empirical coverage.
    """

    coverage = float(np.mean(covered))
    r = covered.shape[0]
    return float(np.sqrt(coverage * (1.0 - coverage) / r))
