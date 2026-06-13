"""Formula and smoke-output diagnostics for active linear PLR DGPs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dml_project import config
from dml_project.dgps.base import (
    alternating_dense_coefficients,
    alternating_sparse_coefficients,
    ar1_covariance,
    dense_coefficients,
    sparse_coefficients,
)

SIGMA_U = 1.0
SIGMA_V = 1.0
AR1_RHO = 0.5

DGP_FORMULAS = {
    "dense_linear_independent": {
        "covariance_structure": "independent standard normal",
        "beta_r_formula": "1 / j for j = 1, ..., p",
        "beta_g_formula": "(-1)^(j + 1) / j for j = 1, ..., p",
        "coefficient_kind": "dense",
        "covariance_kind": "independent",
        "scale": 1.0,
    },
    "sparse_linear_independent": {
        "covariance_structure": "independent standard normal",
        "beta_r_formula": "1 / j for j = 1, ..., min(10, p); zero otherwise",
        "beta_g_formula": "(-1)^(j + 1) / j for j = 1, ..., min(10, p); zero otherwise",
        "coefficient_kind": "sparse",
        "covariance_kind": "independent",
        "scale": 1.0,
    },
    "sparse_linear_correlated": {
        "covariance_structure": "AR(1) Gaussian with rho = 0.5",
        "beta_r_formula": "1 / j for j = 1, ..., min(10, p); zero otherwise",
        "beta_g_formula": "(-1)^(j + 1) / j for j = 1, ..., min(10, p); zero otherwise",
        "coefficient_kind": "sparse",
        "covariance_kind": "ar1",
        "scale": 1.0,
    },
    "weak_signal_sparse": {
        "covariance_structure": "independent standard normal",
        "beta_r_formula": "0.25 / j for j = 1, ..., min(10, p); zero otherwise",
        "beta_g_formula": "0.25 * (-1)^(j + 1) / j for j = 1, ..., min(10, p); zero otherwise",
        "coefficient_kind": "sparse",
        "covariance_kind": "independent",
        "scale": 0.25,
    },
}

DGP_LEARNER_COLUMNS = [
    "row_type",
    "dgp_name",
    "learner_name",
    "scenario_name",
    "n_obs",
    "n_covariates",
    "n_folds",
    "mean_mae",
    "mean_rmse",
    "mean_coverage",
    "mean_se_ratio",
    "mean_nuisance_r2_y",
    "mean_nuisance_r2_d",
    "mean_fold_ratio",
    "mean_condition_number",
    "rank_deficiency_rate",
    "mae",
    "rmse",
    "coverage",
    "se_ratio",
]

SE_RATIO_COLUMNS = [
    "dgp_name",
    "mean_se_ratio",
    "mean_coverage",
    "mean_ci_length",
    "mean_empirical_sd",
    "diagnosis",
]


def coefficient_vectors_for_dgp(dgp_name: str, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Return treatment and outcome nuisance coefficients for an active DGP."""

    if dgp_name not in DGP_FORMULAS:
        raise ValueError(f"Unknown DGP: {dgp_name}")
    spec = DGP_FORMULAS[dgp_name]
    scale = float(spec["scale"])
    if spec["coefficient_kind"] == "dense":
        return (
            dense_coefficients(n_covariates=p, scale=scale),
            alternating_dense_coefficients(n_covariates=p, scale=scale),
        )
    return (
        sparse_coefficients(n_covariates=p, n_active=10, scale=scale),
        alternating_sparse_coefficients(n_covariates=p, n_active=10, scale=scale),
    )


def covariance_matrix_for_dgp(dgp_name: str, p: int) -> np.ndarray:
    """Return the X covariance matrix for an active DGP."""

    if dgp_name not in DGP_FORMULAS:
        raise ValueError(f"Unknown DGP: {dgp_name}")
    if DGP_FORMULAS[dgp_name]["covariance_kind"] == "ar1":
        return ar1_covariance(n_covariates=p, rho=AR1_RHO)
    return np.eye(p)


def linear_variance(beta: np.ndarray, covariance: np.ndarray) -> float:
    """Compute Var(X beta) = beta' Sigma beta."""

    return float(beta.T @ covariance @ beta)


def formula_audit_table(p_values: list[int] | None = None) -> pd.DataFrame:
    """Return exact formula and signal-strength diagnostics for active DGPs."""

    rows: list[dict[str, object]] = []
    for dgp_name in config.DGP_NAMES:
        spec = DGP_FORMULAS[dgp_name]
        for p in p_values or config.P_VALUES:
            beta_r, beta_g = coefficient_vectors_for_dgp(dgp_name, p)
            covariance = covariance_matrix_for_dgp(dgp_name, p)
            var_r0 = linear_variance(beta_r, covariance)
            var_g0 = linear_variance(beta_g, covariance)
            var_v = SIGMA_V**2
            var_u = SIGMA_U**2
            var_d = var_r0 + var_v
            rows.append(
                {
                    "dgp_name": dgp_name,
                    "p": p,
                    "covariance_structure": spec["covariance_structure"],
                    "beta_r_formula": spec["beta_r_formula"],
                    "beta_g_formula": spec["beta_g_formula"],
                    "active_dimension": int(np.count_nonzero(beta_r)),
                    "beta_r_norm": float(np.linalg.norm(beta_r)),
                    "beta_g_norm": float(np.linalg.norm(beta_g)),
                    "var_r0": var_r0,
                    "var_g0": var_g0,
                    "var_v": var_v,
                    "var_u": var_u,
                    "var_d": var_d,
                    "var_r0_over_var_v": var_r0 / var_v,
                    "var_g0_over_var_u": var_g0 / var_u,
                    "var_r0_over_var_d": var_r0 / var_d,
                }
            )
    return pd.DataFrame(rows)


def _read_summary_if_available(path: Path) -> pd.DataFrame | None:
    """Read a smoke summary if available, normalizing n and p aliases."""

    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "n_obs" not in df.columns and "n" in df.columns:
        df["n_obs"] = df["n"]
    if "n_covariates" not in df.columns and "p" in df.columns:
        df["n_covariates"] = df["p"]
    return df


def _optional_mean_columns(df: pd.DataFrame, columns: list[str]) -> dict[str, tuple[str, str]]:
    """Return named aggregation specs for columns present in a DataFrame."""

    return {f"mean_{column}": (column, "mean") for column in columns if column in df.columns}


def worst_dgp_diagnosis_table(summary_df: pd.DataFrame | None) -> pd.DataFrame:
    """Return DGP-learner summaries and worst scenarios when smoke output exists."""

    if summary_df is None:
        return pd.DataFrame(columns=DGP_LEARNER_COLUMNS)
    required = {"dgp_name", "learner_name", "mae", "rmse", "coverage"}
    if missing := required.difference(summary_df.columns):
        raise ValueError(f"Scenario summary missing required columns: {sorted(missing)}")

    mean_columns = [
        "mae",
        "rmse",
        "coverage",
        "se_ratio",
        "nuisance_r2_y",
        "nuisance_r2_d",
        "mean_nuisance_r2_y",
        "mean_nuisance_r2_d",
        "mean_fold_ratio",
        "mean_condition_number",
        "rank_deficiency_rate",
    ]
    summary = (
        summary_df.groupby(["dgp_name", "learner_name"], as_index=False)
        .agg(**_optional_mean_columns(summary_df, mean_columns))
        .rename(
            columns={
                "mean_mae": "mean_mae",
                "mean_rmse": "mean_rmse",
                "mean_coverage": "mean_coverage",
                "mean_se_ratio": "mean_se_ratio",
                "mean_mean_nuisance_r2_y": "mean_nuisance_r2_y",
                "mean_mean_nuisance_r2_d": "mean_nuisance_r2_d",
                "mean_mean_fold_ratio": "mean_fold_ratio",
                "mean_mean_condition_number": "mean_condition_number",
                "mean_rank_deficiency_rate": "rank_deficiency_rate",
            }
        )
    )
    summary["row_type"] = "dgp_learner_summary"
    for column in DGP_LEARNER_COLUMNS:
        if column not in summary.columns:
            summary[column] = np.nan

    worst_rows = []
    for dgp_name, dgp_df in summary_df.groupby("dgp_name", sort=True):
        worst = dgp_df.sort_values(
            ["coverage", "mae", "rmse"],
            ascending=[True, False, False],
            kind="mergesort",
        ).head(10)
        worst = worst.copy()
        worst["row_type"] = "worst_scenario"
        worst_rows.append(worst)
    worst_table = pd.concat(worst_rows, ignore_index=True) if worst_rows else pd.DataFrame()
    for column in DGP_LEARNER_COLUMNS:
        if column not in worst_table.columns:
            worst_table[column] = np.nan

    return pd.concat(
        [
            summary.loc[:, DGP_LEARNER_COLUMNS],
            worst_table.loc[:, DGP_LEARNER_COLUMNS],
        ],
        ignore_index=True,
    )


def se_ratio_by_dgp_table(summary_df: pd.DataFrame | None) -> pd.DataFrame:
    """Return SE-ratio and coverage diagnostics by DGP when smoke output exists."""

    if summary_df is None:
        return pd.DataFrame(columns=SE_RATIO_COLUMNS)
    required = {"dgp_name", "coverage"}
    if missing := required.difference(summary_df.columns):
        raise ValueError(f"Scenario summary missing required columns: {sorted(missing)}")

    aggregations = {"mean_coverage": ("coverage", "mean")}
    for output_column, input_column in [
        ("mean_se_ratio", "se_ratio"),
        ("mean_ci_length", "ci_length"),
        ("mean_empirical_sd", "empirical_sd"),
    ]:
        if input_column in summary_df.columns:
            aggregations[output_column] = (input_column, "mean")
    table = summary_df.groupby("dgp_name", as_index=False).agg(**aggregations)
    for column in SE_RATIO_COLUMNS:
        if column not in table.columns:
            table[column] = np.nan
    table["diagnosis"] = np.where(
        (table["mean_coverage"] < 0.95) & (table["mean_se_ratio"] < 1.0),
        "coverage below 0.95 with se_ratio below 1: intervals likely too narrow",
        "no narrow-interval signal from aggregate se_ratio",
    )
    return table.loc[:, SE_RATIO_COLUMNS]


def create_dgp_diagnostic_reports(
    output_dir: str | Path = config.OUTPUT_DIR / "diagnostics",
    scenario_summary_path: str | Path = config.AGGREGATED_RESULTS_DIR / "scenario_summary_smoke.csv",
) -> dict[str, pd.DataFrame]:
    """Create formula and optional smoke-result DGP diagnostics as CSV files."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary_df = _read_summary_if_available(Path(scenario_summary_path))

    formula = formula_audit_table()
    signal_columns = [
        "dgp_name",
        "p",
        "active_dimension",
        "beta_r_norm",
        "beta_g_norm",
        "var_r0",
        "var_g0",
        "var_r0_over_var_v",
        "var_g0_over_var_u",
        "var_v",
        "var_d",
        "var_r0_over_var_d",
    ]
    tables = {
        "formula_audit": formula,
        "dgp_signal_strength": formula.loc[:, signal_columns],
        "worst_dgp_diagnosis": worst_dgp_diagnosis_table(summary_df),
        "se_ratio_by_dgp": se_ratio_by_dgp_table(summary_df),
    }
    for name, table in tables.items():
        table.to_csv(destination / f"{name}.csv", index=False, encoding="utf-8")
    return tables


def main() -> None:
    """CLI entry point for generating DGP diagnostic reports."""

    create_dgp_diagnostic_reports()


if __name__ == "__main__":
    main()
