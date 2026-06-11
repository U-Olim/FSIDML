"""Aggregate replication-level simulation outputs into scenario summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dml_project.simulation.metrics import mc_se_bias, mc_se_coverage, mc_se_rmse

SCENARIO_GROUP_COLUMNS = [
    "scenario_name",
    "n_obs",
    "n_covariates",
    "n_folds",
    "dgp_name",
    "learner_name",
]

DIAGNOSTIC_COLUMNS = [
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

MAX_DIAGNOSTIC_COLUMNS = {"max_fold_ratio", "max_condition_number"}
MIN_DIAGNOSTIC_COLUMNS = {"min_min_eigenvalue"}


def _safe_mean(values: pd.Series | np.ndarray) -> float:
    """Return a NaN-aware mean or NaN when no finite value exists."""

    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _safe_median(values: pd.Series | np.ndarray) -> float:
    """Return a NaN-aware median or NaN when no finite value exists."""

    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _safe_std(values: pd.Series | np.ndarray) -> float:
    """Return sample standard deviation when at least two finite values exist."""

    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size < 2:
        return float("nan")
    return float(np.std(finite, ddof=1))


def _safe_extreme(values: pd.Series | np.ndarray, op: str) -> float:
    """Return a NaN-aware min or max."""

    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    if op == "max":
        return float(np.max(finite))
    if op == "min":
        return float(np.min(finite))
    raise ValueError(f"Unknown extreme op: {op}")


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return numerator / denominator when the denominator is usable."""

    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0.0:
        return float("nan")
    return float(numerator / denominator)


def _prepare_replications(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize old and new replication outputs for aggregation."""

    prepared = df.copy()
    if "theta_0" not in prepared.columns:
        if "theta_true" not in prepared.columns:
            raise ValueError("Input DataFrame must include theta_0 or theta_true")
        prepared["theta_0"] = prepared["theta_true"]
    if "theta_true" not in prepared.columns:
        prepared["theta_true"] = prepared["theta_0"]

    if "failed" not in prepared.columns:
        prepared["failed"] = False
    prepared["failed"] = prepared["failed"].fillna(False).astype(bool)

    if "covered" not in prepared.columns:
        prepared["covered"] = (
            (prepared["ci_lower"] <= prepared["theta_0"])
            & (prepared["theta_0"] <= prepared["ci_upper"])
        )

    if "ci_length" not in prepared.columns:
        prepared["ci_length"] = prepared["ci_upper"] - prepared["ci_lower"]

    if "t_stat" not in prepared.columns:
        prepared["t_stat"] = np.where(
            prepared["se"] > 0,
            (prepared["theta_hat"] - prepared["theta_0"]) / prepared["se"],
            np.nan,
        )

    return prepared


def _successful_rows(group: pd.DataFrame) -> pd.DataFrame:
    """Return rows that should contribute to performance metrics."""

    return group.loc[(~group["failed"]) & np.isfinite(group["theta_hat"])]


def _group_columns(df: pd.DataFrame) -> list[str]:
    """Choose available scenario identifier columns for aggregation."""

    columns = [column for column in SCENARIO_GROUP_COLUMNS if column in df.columns]
    if "scenario_id" in df.columns:
        columns = ["scenario_id", *columns]
    if not columns:
        raise ValueError("Input DataFrame must include scenario_id or scenario metadata")
    return columns


def _diagnostic_value(successful: pd.DataFrame, column: str) -> float:
    """Aggregate one diagnostic column over successful replications."""

    if column not in successful.columns or successful.empty:
        return float("nan")
    if column in MAX_DIAGNOSTIC_COLUMNS:
        return _safe_extreme(successful[column], "max")
    if column in MIN_DIAGNOSTIC_COLUMNS:
        return _safe_extreme(successful[column], "min")
    return _safe_mean(successful[column])


def _aggregate_group(group_key, group_columns: list[str], group: pd.DataFrame) -> dict:
    """Aggregate one scenario group."""

    if not isinstance(group_key, tuple):
        group_key = (group_key,)

    successful = _successful_rows(group)
    n_total = int(len(group))
    n_success = int(len(successful))
    n_failed = int(group["failed"].sum())
    theta_0 = _safe_mean(group["theta_0"])

    row: dict[str, float | int | str] = dict(zip(group_columns, group_key))
    row.update(
        {
            "theta_0": theta_0,
            "theta_true": theta_0,
            "n_replications_total": n_total,
            "n_replications_success": n_success,
            "n_replications_failed": n_failed,
            "non_convergence_rate": float(n_failed / n_total) if n_total > 0 else np.nan,
        }
    )

    if n_success == 0:
        row.update(
            {
                "bias": np.nan,
                "median_bias": np.nan,
                "mae": np.nan,
                "rmse": np.nan,
                "coverage": np.nan,
                "ci_length": np.nan,
                "mean_ci_length": np.nan,
                "mean_se": np.nan,
                "empirical_sd": np.nan,
                "se_ratio": np.nan,
                "variance_ratio": np.nan,
                "mc_se_bias": np.nan,
                "mc_se_rmse": np.nan,
                "mc_se_coverage": np.nan,
                "t_stat_mean": np.nan,
                "t_stat_sd": np.nan,
            }
        )
    else:
        errors = successful["theta_hat"].to_numpy(dtype=float) - successful[
            "theta_0"
        ].to_numpy(dtype=float)
        theta_hat = successful["theta_hat"].to_numpy(dtype=float)
        theta_true = successful["theta_0"].to_numpy(dtype=float)
        covered = successful["covered"].astype(bool).to_numpy(dtype=int)
        mean_se = _safe_mean(successful["se"])
        empirical_sd = _safe_std(successful["theta_hat"])
        row.update(
            {
                "bias": _safe_mean(errors),
                "median_bias": _safe_median(errors),
                "mae": _safe_mean(np.abs(errors)),
                "rmse": float(np.sqrt(_safe_mean(errors**2))),
                "coverage": _safe_mean(covered),
                "ci_length": _safe_mean(successful["ci_length"]),
                "mean_ci_length": _safe_mean(successful["ci_length"]),
                "mean_se": mean_se,
                "empirical_sd": empirical_sd,
                "se_ratio": _safe_ratio(mean_se, empirical_sd),
                "variance_ratio": _safe_ratio(empirical_sd, mean_se),
                "mc_se_bias": (
                    mc_se_bias(theta_hat=theta_hat, theta_true=theta_true)
                    if n_success >= 2
                    else np.nan
                ),
                "mc_se_rmse": (
                    mc_se_rmse(theta_hat=theta_hat, theta_true=theta_true)
                    if n_success >= 2
                    else np.nan
                ),
                "mc_se_coverage": mc_se_coverage(covered),
                "t_stat_mean": _safe_mean(successful["t_stat"]),
                "t_stat_sd": _safe_std(successful["t_stat"]),
            }
        )

    for column in DIAGNOSTIC_COLUMNS:
        row[column] = _diagnostic_value(successful, column)

    return row


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate replication-level outputs to scenario-level summaries.

    Failed replications are counted in non-convergence metrics but excluded from
    point-estimation, inference, and diagnostic summaries.
    """

    prepared = _prepare_replications(df)
    group_columns = _group_columns(prepared)
    rows = [
        _aggregate_group(group_key, group_columns, group)
        for group_key, group in prepared.groupby(
            group_columns,
            sort=True,
            dropna=False,
        )
    ]
    return pd.DataFrame(rows)
