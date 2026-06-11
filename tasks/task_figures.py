"""Pytask step and helpers for publication-ready simulation figures."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
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

NONLINEAR_DGPS = [
    "quadratic_confounding",
    "interaction_confounding",
    "step_confounding",
]

SORT_COLUMNS = ["dgp_name", "learner_name", "n_obs", "n_covariates", "n_folds"]

LEARNER_COLORS = {
    "ols": "#4c78a8",
    "lasso": "#f58518",
    "elastic_net": "#54a24b",
    "random_forest": "#b279a2",
    "gradient_boosting": "#e45756",
}

LEARNER_MARKERS = {
    "ols": "o",
    "lasso": "s",
    "elastic_net": "^",
    "random_forest": "D",
    "gradient_boosting": "v",
}

DGP_MARKERS = {
    "linear_confounding": "o",
    "quadratic_confounding": "s",
    "interaction_confounding": "^",
    "step_confounding": "D",
}

METRIC_LABELS = {
    "coverage": "Coverage",
    "mae": "MAE",
    "rmse": "RMSE",
    "ci_length": "CI length",
    "se_ratio": "SE ratio",
    "non_convergence_rate": "Non-convergence rate",
}

K_SENSITIVITY_METRICS = set(METRIC_LABELS)
NONLINEAR_METRICS = {"mae", "rmse", "coverage", "non_convergence_rate"}


def _label_dgp(dgp_name: str) -> str:
    """Return a readable DGP label."""

    return DGP_LABELS.get(dgp_name, dgp_name)


def _label_learner(learner_name: str) -> str:
    """Return a readable learner label."""

    return LEARNER_LABELS.get(learner_name, learner_name)


def _require_columns(df: pd.DataFrame, columns: set[str], context: str) -> None:
    """Raise a clear error if required columns are missing."""

    missing = sorted(columns.difference(df.columns))
    if missing:
        raise ValueError(f"{context} requires columns: {missing}")


def _normalize_design_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with n_obs and n_covariates aliases normalized."""

    prepared = df.copy()
    if "n_obs" not in prepared.columns and "n" in prepared.columns:
        prepared["n_obs"] = prepared["n"]
    if "n_covariates" not in prepared.columns and "p" in prepared.columns:
        prepared["n_covariates"] = prepared["p"]
    return prepared


def _sort_results(df: pd.DataFrame) -> pd.DataFrame:
    """Sort results deterministically by available scenario columns."""

    prepared = _normalize_design_columns(df)
    sort_columns = [column for column in SORT_COLUMNS if column in prepared.columns]
    if sort_columns:
        prepared = prepared.sort_values(sort_columns, kind="mergesort")
    return prepared.reset_index(drop=True)


def _maybe_save_figure(fig: Figure, save_path: str | Path | None) -> None:
    """Save a figure when a path is provided."""

    if save_path is None:
        return
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")


def _with_rho_fold(df: pd.DataFrame, context: str) -> pd.DataFrame:
    """Add rho_fold from mean_fold_ratio or scenario dimensions."""

    prepared = _normalize_design_columns(df)
    if "mean_fold_ratio" in prepared.columns:
        prepared["rho_fold"] = pd.to_numeric(prepared["mean_fold_ratio"], errors="coerce")
        if prepared["rho_fold"].notna().all():
            return prepared

    _require_columns(
        prepared,
        {"n_obs", "n_covariates", "n_folds"},
        context,
    )
    n_train = prepared["n_obs"] * (1.0 - 1.0 / prepared["n_folds"])
    computed_rho = prepared["n_covariates"] / n_train
    if "rho_fold" in prepared.columns:
        prepared["rho_fold"] = prepared["rho_fold"].fillna(computed_rho)
    else:
        prepared["rho_fold"] = computed_rho
    return prepared


def _learner_order(df: pd.DataFrame) -> list[str]:
    """Return learners in label-map order, followed by any unknown learners."""

    present = list(dict.fromkeys(df["learner_name"].astype(str)))
    known = [learner for learner in LEARNER_LABELS if learner in present]
    unknown = sorted(learner for learner in present if learner not in LEARNER_LABELS)
    return [*known, *unknown]


def plot_coverage_by_fold_ratio(
    df: pd.DataFrame,
    dgp_name: str | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot coverage against fold-level dimensionality.

    When ``dgp_name`` is omitted, rows are averaged by learner and fold ratio.
    """

    prepared = _with_rho_fold(df, "plot_coverage_by_fold_ratio")
    _require_columns(
        prepared,
        {"coverage", "learner_name", "dgp_name", "rho_fold"},
        "plot_coverage_by_fold_ratio",
    )
    if dgp_name is not None:
        prepared = prepared.loc[prepared["dgp_name"] == dgp_name].copy()
    if prepared.empty:
        raise ValueError("plot_coverage_by_fold_ratio has no rows to plot")

    if dgp_name is None:
        plot_df = (
            prepared.groupby(["learner_name", "rho_fold"], as_index=False, sort=True)["coverage"]
            .mean()
            .sort_values(["learner_name", "rho_fold"])
        )
    else:
        plot_df = prepared.sort_values(["learner_name", "rho_fold"])

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for learner_name in _learner_order(plot_df):
        learner_df = plot_df.loc[plot_df["learner_name"] == learner_name].sort_values("rho_fold")
        ax.plot(
            learner_df["rho_fold"],
            learner_df["coverage"],
            marker=LEARNER_MARKERS.get(learner_name, "o"),
            color=LEARNER_COLORS.get(learner_name, "#333333"),
            linewidth=1.8,
            label=_label_learner(learner_name),
        )

    ax.axhline(0.95, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Fold-level dimensionality, p / n_train")
    ax.set_ylabel("Coverage")
    title = "Coverage by fold-level dimensionality"
    if dgp_name is not None:
        title += f": {_label_dgp(dgp_name)}"
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _maybe_save_figure(fig, save_path)
    return fig


def plot_condition_number_vs_se_distortion(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot conditioning against standard-error distortion."""

    condition_column = (
        "mean_condition_number"
        if "mean_condition_number" in df.columns
        else "max_condition_number"
    )
    _require_columns(
        df,
        {condition_column, "se_ratio", "learner_name", "dgp_name"},
        "plot_condition_number_vs_se_distortion",
    )
    prepared = _sort_results(df)
    prepared = prepared.loc[
        np.isfinite(prepared[condition_column])
        & np.isfinite(prepared["se_ratio"])
        & (prepared[condition_column] > 0)
    ].copy()
    if prepared.empty:
        raise ValueError("plot_condition_number_vs_se_distortion has no finite positive rows")

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for learner_name in _learner_order(prepared):
        learner_df = prepared.loc[prepared["learner_name"] == learner_name]
        ax.scatter(
            learner_df[condition_column],
            learner_df["se_ratio"],
            color=LEARNER_COLORS.get(learner_name, "#333333"),
            marker=LEARNER_MARKERS.get(learner_name, "o"),
            label=_label_learner(learner_name),
            alpha=0.85,
        )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("Mean condition number")
    ax.set_ylabel("SE ratio")
    ax.set_title("Conditioning and standard-error distortion")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _maybe_save_figure(fig, save_path)
    return fig


def plot_k_sensitivity(
    df: pd.DataFrame,
    metric: str = "coverage",
    dgp_name: str | None = None,
    learner_name: str | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot how cross-fitting fold choice changes a metric.

    If ``dgp_name`` is omitted, the metric is averaged over DGPs before plotting.
    """

    if metric not in K_SENSITIVITY_METRICS:
        raise ValueError(f"metric must be one of {sorted(K_SENSITIVITY_METRICS)}")
    _require_columns(
        df,
        {"n_folds", metric, "learner_name", "dgp_name"},
        "plot_k_sensitivity",
    )
    prepared = _sort_results(df)
    if dgp_name is not None:
        prepared = prepared.loc[prepared["dgp_name"] == dgp_name].copy()
    if learner_name is not None:
        prepared = prepared.loc[prepared["learner_name"] == learner_name].copy()
    if prepared.empty:
        raise ValueError("plot_k_sensitivity has no rows to plot")

    group_columns = ["n_folds"] if learner_name is not None else ["learner_name", "n_folds"]
    plot_df = (
        prepared.groupby(group_columns, as_index=False, sort=True)[metric]
        .mean()
        .sort_values(group_columns)
    )

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    if learner_name is not None:
        ax.plot(
            plot_df["n_folds"],
            plot_df[metric],
            marker=LEARNER_MARKERS.get(learner_name, "o"),
            color=LEARNER_COLORS.get(learner_name, "#333333"),
            linewidth=1.8,
            label=_label_learner(learner_name),
        )
    else:
        for current_learner in _learner_order(plot_df):
            learner_df = plot_df.loc[plot_df["learner_name"] == current_learner]
            ax.plot(
                learner_df["n_folds"],
                learner_df[metric],
                marker=LEARNER_MARKERS.get(current_learner, "o"),
                color=LEARNER_COLORS.get(current_learner, "#333333"),
                linewidth=1.8,
                label=_label_learner(current_learner),
            )

    if metric == "coverage":
        ax.axhline(0.95, color="black", linestyle="--", linewidth=1.0)
    elif metric == "se_ratio":
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Number of folds K")
    ax.set_ylabel(METRIC_LABELS[metric])
    title = "Sensitivity to cross-fitting folds"
    if dgp_name is not None:
        title += f": {_label_dgp(dgp_name)}"
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _maybe_save_figure(fig, save_path)
    return fig


def plot_nonlinear_dgp_learners(
    df: pd.DataFrame,
    metric: str = "mae",
    save_path: str | Path | None = None,
) -> Figure:
    """Plot learner performance in nonlinear DGPs as grouped bars."""

    if metric not in NONLINEAR_METRICS:
        raise ValueError(f"metric must be one of {sorted(NONLINEAR_METRICS)}")
    _require_columns(
        df,
        {"dgp_name", "learner_name", metric},
        "plot_nonlinear_dgp_learners",
    )
    prepared = _sort_results(df)
    prepared = prepared.loc[prepared["dgp_name"].isin(NONLINEAR_DGPS)].copy()
    if prepared.empty:
        raise ValueError("plot_nonlinear_dgp_learners has no nonlinear DGP rows")

    plot_df = (
        prepared.groupby(["dgp_name", "learner_name"], as_index=False, sort=True)[metric]
        .mean()
        .sort_values(["dgp_name", "learner_name"])
    )
    dgp_order = [dgp for dgp in NONLINEAR_DGPS if dgp in set(plot_df["dgp_name"])]
    learner_order = _learner_order(plot_df)
    x_positions = np.arange(len(dgp_order), dtype=float)
    width = 0.8 / max(len(learner_order), 1)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for learner_index, learner in enumerate(learner_order):
        offsets = x_positions - 0.4 + width / 2 + learner_index * width
        values = []
        for dgp in dgp_order:
            match = plot_df.loc[
                (plot_df["dgp_name"] == dgp) & (plot_df["learner_name"] == learner),
                metric,
            ]
            values.append(float(match.iloc[0]) if len(match) else np.nan)
        ax.bar(
            offsets,
            values,
            width=width,
            color=LEARNER_COLORS.get(learner, "#333333"),
            label=_label_learner(learner),
        )

    if metric == "coverage":
        ax.axhline(0.95, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(x_positions, [_label_dgp(dgp) for dgp in dgp_order])
    ax.set_xlabel("DGP")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title("Learner performance in nonlinear DGPs")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=2)
    _maybe_save_figure(fig, save_path)
    return fig


def task_figures(
    path_to_aggregated_results: Path = (
        config.AGGREGATED_RESULTS_DIR / f"scenario_summary{SUFFIX}.csv"
    ),
    path_to_figure_manifest: Annotated[Path, Product] = (
        PROJECT_ROOT / f"documents/outputs/figures/figure_manifest{SUFFIX}.txt"
    ),
) -> None:
    """Create the revised main simulation figure suite."""

    expected_results = config.AGGREGATED_RESULTS_DIR / f"scenario_summary{SUFFIX}.csv"
    if path_to_aggregated_results != expected_results:
        raise ValueError(f"task_figures must read {expected_results}, got {path_to_aggregated_results}")

    results = pd.read_csv(path_to_aggregated_results)
    figure_dir = config.FIGURES_DIR
    figure_dir.mkdir(parents=True, exist_ok=True)

    output_paths = [
        figure_dir / f"figure_coverage_by_fold_ratio{SUFFIX}.png",
        figure_dir / f"figure_condition_number_vs_se_distortion{SUFFIX}.png",
        figure_dir / f"figure_k_sensitivity{SUFFIX}.png",
        figure_dir / f"figure_nonlinear_dgp_learners{SUFFIX}.png",
    ]
    figures = [
        plot_coverage_by_fold_ratio(results, save_path=output_paths[0]),
        plot_condition_number_vs_se_distortion(results, save_path=output_paths[1]),
        plot_k_sensitivity(results, metric="coverage", save_path=output_paths[2]),
        plot_nonlinear_dgp_learners(results, metric="mae", save_path=output_paths[3]),
    ]
    for fig in figures:
        plt.close(fig)

    manifest_lines = [str(path.relative_to(PROJECT_ROOT).as_posix()) for path in output_paths]
    path_to_figure_manifest.parent.mkdir(parents=True, exist_ok=True)
    path_to_figure_manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
