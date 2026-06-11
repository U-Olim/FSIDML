"""Pytask step and helpers for publication-quality simulation tables."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from pytask import Product

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    from dml_project import config
    from dml_project.pipeline_mode import get_run_mode, output_suffix
except ModuleNotFoundError:
    from src.dml_project import config
    from src.dml_project.pipeline_mode import get_run_mode, output_suffix

MODE = get_run_mode()
SUFFIX = output_suffix(MODE)

DGP_LABELS = {
    "linear_confounding": "Linear",
    "quadratic_confounding": "Quadratic",
    "interaction_confounding": "Interaction",
    "step_confounding": "Step",
}

LEARNER_LABELS = {
    "ols": "OLS",
    "lasso": "Lasso",
    "elastic_net": "Elastic Net",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}

SORT_COLUMNS = ["dgp_name", "learner_name", "n_obs", "n_covariates", "n_folds"]

CORE_COLUMNS = ["n_obs", "n_covariates", "n_folds", "dgp_name", "learner_name"]

MAIN_METRIC_COLUMNS = [
    "bias",
    "median_bias",
    "mae",
    "rmse",
    "coverage",
    "ci_length",
    "se_ratio",
    "non_convergence_rate",
]

DIAGNOSTIC_COLUMNS = [
    "mean_condition_number",
    "max_condition_number",
    "mean_min_eigenvalue",
    "rank_deficiency_rate",
    "mean_nuisance_mse_y",
    "mean_nuisance_mse_d",
    "mean_nuisance_r2_y",
    "mean_nuisance_r2_d",
]


def _label_dgp(dgp_name: str) -> str:
    """Return a publication label for a DGP name."""

    return DGP_LABELS.get(dgp_name, dgp_name)


def _label_learner(learner_name: str) -> str:
    """Return a publication label for a learner name."""

    return LEARNER_LABELS.get(learner_name, learner_name)


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a copy with optional columns added as NaN."""

    prepared = df.copy()
    if "n_obs" not in prepared.columns and "n" in prepared.columns:
        prepared["n_obs"] = prepared["n"]
    if "n_covariates" not in prepared.columns and "p" in prepared.columns:
        prepared["n_covariates"] = prepared["p"]

    missing_core = [column for column in CORE_COLUMNS if column not in prepared.columns]
    if missing_core:
        raise ValueError(f"Missing required table identifier columns: {missing_core}")

    for column in columns:
        if column not in prepared.columns:
            prepared[column] = np.nan
    return prepared


def _sorted_results(df: pd.DataFrame) -> pd.DataFrame:
    """Sort aggregated results deterministically by scenario design."""

    return df.sort_values(SORT_COLUMNS).reset_index(drop=True)


def _compute_design_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add n_train and rho_fold using diagnostics when available."""

    prepared = df.copy()
    computed_n_train = prepared["n_obs"] * (1.0 - 1.0 / prepared["n_folds"])
    if "mean_fold_train_size" in prepared.columns:
        prepared["n_train"] = prepared["mean_fold_train_size"].fillna(computed_n_train)
    else:
        prepared["n_train"] = computed_n_train

    computed_rho = prepared["n_covariates"] / prepared["n_train"]
    if "mean_fold_ratio" in prepared.columns:
        prepared["rho_fold"] = prepared["mean_fold_ratio"].fillna(computed_rho)
    else:
        prepared["rho_fold"] = computed_rho
    return prepared


def _base_table(df: pd.DataFrame, optional_columns: list[str]) -> pd.DataFrame:
    """Prepare common labels, sorting, and design columns."""

    prepared = _ensure_columns(df, optional_columns)
    prepared = _compute_design_columns(_sorted_results(prepared))
    prepared["DGP"] = prepared["dgp_name"].map(_label_dgp)
    prepared["Learner"] = prepared["learner_name"].map(_label_learner)
    prepared["n"] = prepared["n_obs"].astype(int)
    prepared["p"] = prepared["n_covariates"].astype(int)
    prepared["K"] = prepared["n_folds"].astype(int)
    return prepared


def _round_table(df: pd.DataFrame, decimals: dict[str, int]) -> pd.DataFrame:
    """Round selected columns without changing table shape."""

    rounded = df.copy()
    for column, ndigits in decimals.items():
        if column in rounded.columns:
            rounded[column] = pd.to_numeric(rounded[column], errors="coerce").round(ndigits)
    return rounded


def table_simulation_design(aggregated_df: pd.DataFrame) -> pd.DataFrame:
    """Build the simulation design table from aggregated results."""

    prepared = _base_table(
        aggregated_df,
        optional_columns=["mean_fold_train_size", "mean_fold_ratio"],
    )
    table = prepared[["DGP", "Learner", "n", "p", "K", "n_train", "rho_fold"]].drop_duplicates()
    return _round_table(table.reset_index(drop=True), {"n_train": 3, "rho_fold": 3})


def table_main_results(aggregated_df: pd.DataFrame) -> pd.DataFrame:
    """Build the main Monte Carlo performance table."""

    prepared = _base_table(
        aggregated_df,
        optional_columns=["mean_fold_ratio", *MAIN_METRIC_COLUMNS],
    )
    table = prepared[
        [
            "DGP",
            "Learner",
            "n",
            "p",
            "K",
            "rho_fold",
            "bias",
            "median_bias",
            "mae",
            "rmse",
            "coverage",
            "ci_length",
            "se_ratio",
            "non_convergence_rate",
        ]
    ].rename(
        columns={
            "bias": "Bias",
            "median_bias": "Median Bias",
            "mae": "MAE",
            "rmse": "RMSE",
            "coverage": "Coverage",
            "ci_length": "CI Length",
            "se_ratio": "SE Ratio",
            "non_convergence_rate": "Non-convergence",
        }
    )
    return _round_table(
        table,
        {
            "rho_fold": 3,
            "Bias": 3,
            "Median Bias": 3,
            "MAE": 3,
            "RMSE": 3,
            "Coverage": 3,
            "CI Length": 3,
            "SE Ratio": 3,
            "Non-convergence": 3,
        },
    )


def table_diagnostics(aggregated_df: pd.DataFrame) -> pd.DataFrame:
    """Build the diagnostics table for fold design and nuisance quality."""

    prepared = _base_table(
        aggregated_df,
        optional_columns=["mean_fold_ratio", *DIAGNOSTIC_COLUMNS],
    )
    table = prepared[
        [
            "DGP",
            "Learner",
            "n",
            "p",
            "K",
            "rho_fold",
            "mean_condition_number",
            "max_condition_number",
            "mean_min_eigenvalue",
            "rank_deficiency_rate",
            "mean_nuisance_mse_y",
            "mean_nuisance_mse_d",
            "mean_nuisance_r2_y",
            "mean_nuisance_r2_d",
        ]
    ].rename(
        columns={
            "mean_condition_number": "Mean Cond. No.",
            "max_condition_number": "Max Cond. No.",
            "mean_min_eigenvalue": "Mean Min Eigenvalue",
            "rank_deficiency_rate": "Rank Def. Rate",
            "mean_nuisance_mse_y": "Nuisance MSE Y",
            "mean_nuisance_mse_d": "Nuisance MSE D",
            "mean_nuisance_r2_y": "Nuisance R2 Y",
            "mean_nuisance_r2_d": "Nuisance R2 D",
        }
    )
    return _round_table(
        table,
        {
            "rho_fold": 3,
            "Mean Cond. No.": 2,
            "Max Cond. No.": 2,
            "Mean Min Eigenvalue": 4,
            "Rank Def. Rate": 3,
            "Nuisance MSE Y": 3,
            "Nuisance MSE D": 3,
            "Nuisance R2 Y": 3,
            "Nuisance R2 D": 3,
        },
    )


def _to_markdown_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a simple Markdown table."""

    headers = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _write_table_outputs(table: pd.DataFrame, output_stem: Path) -> list[Path]:
    """Write CSV and Markdown outputs for a table."""

    csv_path = output_stem.with_suffix(".csv")
    md_path = output_stem.with_suffix(".md")
    table.to_csv(csv_path, index=False, encoding="utf-8")
    md_path.write_text(_to_markdown_table(table) + "\n", encoding="utf-8")
    return [csv_path, md_path]


def task_tables(
    path_to_raw: Path = config.RAW_RESULTS_DIR / f"simulations{SUFFIX}.csv",
    path_to_summary: Path = config.AGGREGATED_RESULTS_DIR / f"scenario_summary{SUFFIX}.csv",
    path_to_table: Annotated[Path, Product] = (
        config.TABLES_DIR / f"table_main_results{SUFFIX}.csv"
    ),
    path_to_table_manifest: Annotated[Path, Product] = (
        PROJECT_ROOT / f"documents/outputs/tables/table_suite_manifest{SUFFIX}.txt"
    ),
) -> None:
    """Create the revised publication table suite from aggregated results."""

    expected_raw = config.RAW_RESULTS_DIR / f"simulations{SUFFIX}.csv"
    expected_summary = config.AGGREGATED_RESULTS_DIR / f"scenario_summary{SUFFIX}.csv"
    expected_table = config.TABLES_DIR / f"table_main_results{SUFFIX}.csv"
    if path_to_raw != expected_raw:
        raise ValueError(f"task_tables must read {expected_raw}, got {path_to_raw}")
    if path_to_summary != expected_summary:
        raise ValueError(f"task_tables must read {expected_summary}, got {path_to_summary}")
    if path_to_table != expected_table:
        raise ValueError(f"task_tables must write {expected_table}, got {path_to_table}")

    _ = pd.read_csv(path_to_raw)
    aggregated_df = pd.read_csv(path_to_summary)

    table_dir = path_to_table.parent
    table_dir.mkdir(parents=True, exist_ok=True)

    design_table = table_simulation_design(aggregated_df)
    main_table = table_main_results(aggregated_df)
    diagnostics_table = table_diagnostics(aggregated_df)

    generated: list[Path] = []
    generated.extend(_write_table_outputs(design_table, table_dir / f"table_simulation_design{SUFFIX}"))
    generated.extend(_write_table_outputs(main_table, table_dir / f"table_main_results{SUFFIX}"))
    generated.extend(_write_table_outputs(diagnostics_table, table_dir / f"table_diagnostics{SUFFIX}"))

    manifest_lines = [
        str(path.relative_to(PROJECT_ROOT).as_posix())
        for path in sorted(set(generated), key=lambda path: str(path))
    ]
    path_to_table_manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
