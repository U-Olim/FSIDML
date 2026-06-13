"""Diagnostic ranking tables for aggregated simulation results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "scenario_name",
    "dgp_name",
    "learner_name",
    "n_obs",
    "n_covariates",
    "n_folds",
    "mae",
    "rmse",
    "coverage",
]

OPTIONAL_COLUMNS = [
    "bias",
    "ci_length",
    "se_ratio",
    "non_convergence_rate",
    "mean_nuisance_r2_y",
    "mean_nuisance_r2_d",
    "mean_fold_ratio",
    "mean_condition_number",
    "rank_deficiency_rate",
]

AGGREGATION_COLUMNS = [
    "bias",
    "mae",
    "rmse",
    "coverage",
    "ci_length",
    "se_ratio",
    "non_convergence_rate",
    "mean_nuisance_r2_y",
    "mean_nuisance_r2_d",
    "mean_fold_ratio",
]

SUMMARY_METRIC_COLUMNS = [
    "mean_bias",
    "mean_mae",
    "mean_rmse",
    "mean_coverage",
    "mean_ci_length",
    "mean_se_ratio",
    "mean_non_convergence_rate",
    "mean_nuisance_r2_y",
    "mean_nuisance_r2_d",
    "mean_fold_ratio",
    "n_scenarios",
]

SCENARIO_COLUMNS = [
    "scenario_name",
    "dgp_name",
    "learner_name",
    "n_obs",
    "n_covariates",
    "n_folds",
    "mean_fold_ratio",
    "bias",
    "mae",
    "rmse",
    "coverage",
    "ci_length",
    "se_ratio",
    "non_convergence_rate",
    "mean_condition_number",
    "rank_deficiency_rate",
    "mean_nuisance_r2_y",
    "mean_nuisance_r2_d",
]

FOLD_RATIO_BIN_LABELS = [
    "rho <= 0.25",
    "0.25 < rho <= 0.5",
    "0.5 < rho <= 1.0",
    "rho > 1.0",
]


def _read_and_prepare(path: Path) -> pd.DataFrame:
    """Read a scenario summary and normalize expected design columns."""

    df = pd.read_csv(path)
    if "n_obs" not in df.columns and "n" in df.columns:
        df["n_obs"] = df["n"]
    if "n_covariates" not in df.columns and "p" in df.columns:
        df["n_covariates"] = df["p"]

    missing = sorted(set(REQUIRED_COLUMNS).difference(df.columns))
    if missing:
        raise ValueError(f"Scenario summary missing required columns: {missing}")

    prepared = df.copy()
    for column in OPTIONAL_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = np.nan
    return prepared


def _summary_by(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Return mean diagnostics grouped by the supplied columns."""

    grouped = df.groupby(group_columns, dropna=False, sort=True)
    summary = grouped[AGGREGATION_COLUMNS].mean().reset_index()
    summary = summary.rename(
        columns={
            "bias": "mean_bias",
            "mae": "mean_mae",
            "rmse": "mean_rmse",
            "coverage": "mean_coverage",
            "ci_length": "mean_ci_length",
            "se_ratio": "mean_se_ratio",
            "non_convergence_rate": "mean_non_convergence_rate",
        }
    )
    counts = grouped.size().rename("n_scenarios").reset_index()
    return summary.merge(counts, on=group_columns, how="left")


def _learner_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Rank learners by mean MAE and coverage calibration."""

    table = _summary_by(df, ["learner_name"])
    table["_coverage_distance"] = (table["mean_coverage"] - 0.95).abs()
    table = table.sort_values(
        ["mean_mae", "_coverage_distance", "learner_name"],
        kind="mergesort",
    )
    return table.drop(columns=["_coverage_distance"]).loc[
        :,
        ["learner_name", *SUMMARY_METRIC_COLUMNS],
    ].reset_index(drop=True)


def _dgp_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Rank DGPs by mean MAE."""

    table = _summary_by(df, ["dgp_name"])
    return table.sort_values(["mean_mae", "dgp_name"], kind="mergesort").loc[
        :,
        ["dgp_name", *SUMMARY_METRIC_COLUMNS],
    ].reset_index(drop=True)


def _k_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize diagnostics by cross-fitting fold count."""

    table = _summary_by(df, ["n_folds"])
    return table.sort_values("n_folds", kind="mergesort").loc[
        :,
        ["n_folds", *SUMMARY_METRIC_COLUMNS],
    ].reset_index(drop=True)


def _fold_ratio_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize key metrics by fold-ratio bins."""

    bins = [-np.inf, 0.25, 0.5, 1.0, np.inf]
    prepared = df.copy()
    prepared["fold_ratio_bin"] = pd.cut(
        prepared["mean_fold_ratio"],
        bins=bins,
        labels=FOLD_RATIO_BIN_LABELS,
        right=True,
    )
    grouped = prepared.groupby("fold_ratio_bin", observed=False, sort=True)
    table = (
        grouped.agg(
            n_scenarios=("scenario_name", "size"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_coverage=("coverage", "mean"),
            mean_se_ratio=("se_ratio", "mean"),
            mean_non_convergence_rate=("non_convergence_rate", "mean"),
        )
        .reset_index()
    )
    return table


def _worst_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    """Return the 20 weakest scenario-level rows."""

    return (
        df.sort_values(
            ["coverage", "mae", "rmse"],
            ascending=[True, False, False],
            kind="mergesort",
        )
        .head(20)
        .loc[:, SCENARIO_COLUMNS]
        .reset_index(drop=True)
    )


def _best_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    """Return the 20 strongest scenario-level rows."""

    prepared = df.copy()
    prepared["_coverage_distance"] = (prepared["coverage"] - 0.95).abs()
    return (
        prepared.sort_values(
            ["mae", "_coverage_distance", "rmse"],
            ascending=[True, True, True],
            kind="mergesort",
        )
        .head(20)
        .loc[:, SCENARIO_COLUMNS]
        .reset_index(drop=True)
    )


def _rf_diagnostic(df: pd.DataFrame) -> pd.DataFrame:
    """Return random-forest diagnostics by design cell."""

    rf = df.loc[df["learner_name"] == "random_forest"].copy()
    if rf.empty:
        return pd.DataFrame(
            columns=[
                "dgp_name",
                "n_obs",
                "n_covariates",
                "n_folds",
                "mean_mae",
                "mean_rmse",
                "mean_coverage",
                "mean_se_ratio",
                "mean_nuisance_r2_y",
                "mean_nuisance_r2_d",
                "n_scenarios",
            ]
        )
    table = _summary_by(rf, ["dgp_name", "n_obs", "n_covariates", "n_folds"])
    return table.loc[
        :,
        [
            "dgp_name",
            "n_obs",
            "n_covariates",
            "n_folds",
            "mean_mae",
            "mean_rmse",
            "mean_coverage",
            "mean_se_ratio",
            "mean_nuisance_r2_y",
            "mean_nuisance_r2_d",
            "n_scenarios",
        ],
    ].reset_index(drop=True)


def _coverage_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Return scenario-level coverage gaps from the nominal 95 percent target."""

    table = df.copy()
    table["coverage_gap"] = table["coverage"] - 0.95
    table["absolute_coverage_gap"] = table["coverage_gap"].abs()
    return (
        table.sort_values("absolute_coverage_gap", ascending=False, kind="mergesort")
        .loc[
            :,
            [
                "scenario_name",
                "dgp_name",
                "learner_name",
                "n_obs",
                "n_covariates",
                "n_folds",
                "mean_fold_ratio",
                "coverage",
                "coverage_gap",
                "absolute_coverage_gap",
                "se_ratio",
                "mae",
                "rmse",
            ],
        ]
        .reset_index(drop=True)
    )


def create_result_diagnostics(
    scenario_summary_path: str | Path,
    output_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    """Create and save ranking diagnostics from scenario-level summaries."""

    summary_path = Path(scenario_summary_path)
    destination = Path(output_dir)
    df = _read_and_prepare(summary_path)
    suffix = ""
    if summary_path.stem.startswith("scenario_summary"):
        suffix = summary_path.stem.removeprefix("scenario_summary")

    tables = {
        f"learner_ranking{suffix}": _learner_ranking(df),
        f"dgp_ranking{suffix}": _dgp_ranking(df),
        f"k_ranking{suffix}": _k_ranking(df),
        f"fold_ratio_bins{suffix}": _fold_ratio_bins(df),
        f"worst_20_scenarios{suffix}": _worst_scenarios(df),
        f"best_20_scenarios{suffix}": _best_scenarios(df),
        f"rf_diagnostic{suffix}": _rf_diagnostic(df),
        f"coverage_gap{suffix}": _coverage_gap(df),
    }

    destination.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(destination / f"{name}.csv", index=False, encoding="utf-8")
    return tables
