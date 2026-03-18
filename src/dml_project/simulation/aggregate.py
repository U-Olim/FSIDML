"""Aggregate replication-level simulation outputs into scenario summaries.

Process overview:
1. Validate replication-level invariants and copy the input DataFrame.
2. Create missing derived replication columns:
   - ``t_stat`` from ``(theta_hat - theta_true) / se`` when ``se > 0``.
   - ``ci_length`` from ``ci_upper - ci_lower``.
3. Group replications by ``scenario_id``.
4. For each scenario, compute core summaries:
   - point metrics: bias, RMSE, empirical SD, mean SE, coverage, mean CI length
   - uncertainty diagnostics: Monte Carlo SEs for bias/RMSE/coverage
   - calibration diagnostics: variance ratio, t-stat mean, t-stat SD
5. Assemble all scenario summaries into one aggregated DataFrame.
6. Validate aggregate-level invariants before returning results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dml_project.simulation.metrics import (
    bias_and_rmse,
    mc_se_bias,
    mc_se_coverage,
    mc_se_rmse,
)
from dml_project.utils.checks import (
    validate_aggregate_invariants,
    validate_replication_invariants,
)


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate replication-level outputs to metrics by ``scenario_id``.

    Args:
        df: Replication-level results DataFrame. Expected columns include
            ``scenario_id``, ``theta_hat``, ``theta_true``, ``se``, CI bounds,
            and coverage indicators.

    Returns:
        Scenario-level DataFrame with summary metrics such as bias, RMSE,
        coverage, empirical standard deviation, and Monte Carlo standard
        errors for key quantities.

    Raises:
        ValueError: If required aggregated columns are missing.
    """

    validate_replication_invariants(df)
    df = df.copy()
    if "t_stat" not in df.columns:
        df["t_stat"] = np.where(
            df["se"] > 0,
            (df["theta_hat"] - df["theta_true"]) / df["se"],
            np.nan,
        )
    if "ci_length" not in df.columns:
        df["ci_length"] = df["ci_upper"] - df["ci_lower"]
    rows: list[dict[str, float | int]] = []

    for scenario_id, group in df.groupby("scenario_id", sort=True):
        bias, rmse = bias_and_rmse(
            theta_hat=group["theta_hat"].to_numpy(),
            theta_true=group["theta_true"].to_numpy(),
        )
        empirical_sd = float(group["theta_hat"].std(ddof=1))
        mean_se = float(group["se"].mean())
        variance_ratio = empirical_sd / mean_se if mean_se > 0 else np.nan
        t_stat_mean = float(group["t_stat"].mean())
        t_stat_sd = float(group["t_stat"].std(ddof=1))
        scenario_id_value = int(np.asarray(scenario_id).item())
        rows.append(
            {
                "scenario_id": scenario_id_value,
                "bias": bias,
                "mc_se_bias": mc_se_bias(
                    theta_hat=group["theta_hat"].to_numpy(),
                    theta_true=group["theta_true"].to_numpy(),
                ),
                "rmse": rmse,
                "mc_se_rmse": mc_se_rmse(
                    theta_hat=group["theta_hat"].to_numpy(),
                    theta_true=group["theta_true"].to_numpy(),
                ),
                "empirical_sd": empirical_sd,
                "mean_se": mean_se,
                "variance_ratio": variance_ratio,
                "t_stat_mean": t_stat_mean,
                "t_stat_sd": t_stat_sd,
                "coverage": float(group["covered"].mean()),
                "mc_se_coverage": mc_se_coverage(group["covered"].to_numpy()),
                "mean_ci_length": float(group["ci_length"].mean()),
            }
        )

    aggregated = pd.DataFrame(rows)
    required_metric_columns = [
        "bias",
        "rmse",
        "empirical_sd",
        "mean_se",
        "coverage",
        "mean_ci_length",
        "variance_ratio",
        "t_stat_mean",
        "t_stat_sd",
    ]
    for column in required_metric_columns:
        if column not in aggregated.columns:
            raise ValueError(f"Missing required aggregated metric column: {column}")
    validate_aggregate_invariants(raw_df=df, aggregated_df=aggregated)
    return aggregated
