"""Pytask step for figure generation from simulation outputs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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


_LEARNER_LABELS = {"ols": "OLS", "lasso": "Lasso", "elastic_net": "Elastic Net"}
_LEARNER_COLORS = {"ols": "#4c78a8", "lasso": "#f58518", "elastic_net": "#54a24b"}
_LEARNER_MARKERS = {"ols": "o", "lasso": "s", "elastic_net": "^"}
_DGP_LABELS = {
    "linear_baseline": "Linear Baseline",
    "linear_sparse_correlated": "Linear Sparse Correlated",
}
_DGP_FILE_STEMS = {
    "linear_baseline": "linear_baseline",
    "linear_sparse_correlated": "linear_sparse_correlated",
}
_FIGURE_GROUP_KEYS = ["dgp_name", "learner_name", "n", "p", "matched_specification"]


def _prepare_metrics_for_figures(results_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare deterministic design-cell metrics for plotting."""

    required_columns = set(_FIGURE_GROUP_KEYS + ["coverage", "rmse", "bias"])
    missing_columns = sorted(required_columns.difference(results_df.columns))
    if missing_columns:
        raise ValueError(f"main_results missing required plotting columns: {missing_columns}")

    # Figures intentionally preserve n and p design cells. Collapsing over n or p is
    # scientifically invalid for this simulation design and would distort conclusions.
    aggregations: dict[str, tuple[str, str]] = {
        "coverage": ("coverage", "mean"),
        "rmse": ("rmse", "mean"),
        "bias": ("bias", "mean"),
    }
    if "mean_ci_length" in results_df.columns:
        aggregations["mean_ci_length"] = ("mean_ci_length", "mean")

    metrics = results_df.groupby(_FIGURE_GROUP_KEYS, as_index=False, sort=False).agg(**aggregations)

    metrics["dgp_name"] = pd.Categorical(
        metrics["dgp_name"], categories=config.DGP_NAMES, ordered=True
    )
    metrics["p"] = pd.Categorical(metrics["p"], categories=config.P_VALUES, ordered=True)
    metrics["n"] = pd.Categorical(metrics["n"], categories=config.N_VALUES, ordered=True)
    metrics["learner_name"] = pd.Categorical(
        metrics["learner_name"], categories=config.LEARNERS, ordered=True
    )
    metrics = metrics.sort_values(
        ["dgp_name", "p", "n", "learner_name", "matched_specification"],
        kind="mergesort",
    ).reset_index(drop=True)
    metrics["dgp_name"] = metrics["dgp_name"].astype(str)
    metrics["p"] = metrics["p"].astype(int)
    metrics["n"] = metrics["n"].astype(int)
    metrics["learner_name"] = metrics["learner_name"].astype(str)
    metrics["matched_specification"] = metrics["matched_specification"].astype(bool)
    if "mean_ci_length" not in metrics.columns:
        metrics["mean_ci_length"] = np.nan
    metrics["abs_bias"] = metrics["bias"].abs()
    return metrics


def _centered_learner_positions(x_center: float, width: float) -> list[float]:
    """Return symmetric x positions for the 3 learner bars around a center tick."""

    return [x_center + (offset - 1) * width for offset in range(len(config.LEARNERS))]


def _cell_value(
    metrics: pd.DataFrame, dgp_name: str, p_value: int, n_value: int, learner_name: str, metric: str
) -> float:
    """Return one metric value for an exact design cell."""

    cell = metrics.loc[
        (metrics["dgp_name"] == dgp_name)
        & (metrics["p"] == p_value)
        & (metrics["n"] == n_value)
        & (metrics["learner_name"] == learner_name)
    ]
    if len(cell) != 1:
        raise ValueError(
            "Expected exactly one row per plotting cell, got "
            f"{len(cell)} for dgp={dgp_name}, p={p_value}, n={n_value}, learner={learner_name}."
        )
    return float(cell.iloc[0][metric])


def _metric_title(metric_name: str) -> str:
    titles = {
        "coverage": "Empirical 95% CI Coverage",
        "rmse": "RMSE of DML Estimate",
        "bias": "Signed Bias of DML Estimate",
        "abs_bias": "Absolute Bias of DML Estimate",
        "mean_ci_length": "Average 95% CI Length",
    }
    return titles[metric_name]


def _plot_metric_by_n(
    metrics: pd.DataFrame,
    dgp_name: str,
    p_value: int,
    metric_name: str,
    output_path: Path,
    *,
    reference_line: float | None = None,
    zero_line: bool = False,
    y_bounds: tuple[float, float] | None = None,
) -> None:
    """Create a single metric-by-n line figure for a DGP x p design cell."""

    subset = metrics.loc[(metrics["dgp_name"] == dgp_name) & (metrics["p"] == p_value)].copy()
    if subset.empty:
        raise ValueError(f"No plotting data for dgp={dgp_name}, p={p_value}")

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for learner_name in config.LEARNERS:
        learner_subset = subset.loc[subset["learner_name"] == learner_name].sort_values("n")
        ax.plot(
            learner_subset["n"],
            learner_subset[metric_name],
            color=_LEARNER_COLORS[learner_name],
            marker=_LEARNER_MARKERS[learner_name],
            linewidth=2.0,
            markersize=6,
            label=_LEARNER_LABELS[learner_name],
        )

    if reference_line is not None:
        ax.axhline(reference_line, color="black", linewidth=1.1, linestyle="--")
    if zero_line:
        ax.axhline(0.0, color="black", linewidth=1.0)

    ax.set_xlabel("Sample size n")
    ax.set_ylabel(_metric_title(metric_name))
    ax.set_title(f"{_metric_title(metric_name)}: {_DGP_LABELS[dgp_name]}, p={p_value}")
    ax.set_xticks(config.N_VALUES, [str(value) for value in config.N_VALUES])
    if y_bounds is not None:
        ax.set_ylim(*y_bounds)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)

    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_sampling_distribution(
    raw_df: pd.DataFrame,
    dgp_name: str,
    p_value: int,
    requested_n_value: int,
    output_path: Path,
) -> None:
    """Plot replication-level theta_hat distributions by learner for one scenario."""

    subset_all_n = raw_df.loc[(raw_df["dgp_name"] == dgp_name) & (raw_df["p"] == p_value)].copy()
    if subset_all_n.empty:
        raise ValueError(f"No replication data for dgp={dgp_name}, p={p_value}")

    available_n = sorted(int(value) for value in subset_all_n["n"].unique())
    if requested_n_value in available_n:
        actual_n_value = requested_n_value
    else:
        actual_n_value = min(available_n, key=lambda value: abs(value - requested_n_value))

    subset = subset_all_n.loc[subset_all_n["n"] == actual_n_value].copy()

    theta_true = float(subset["theta_true"].iloc[0])
    fig, axes = plt.subplots(1, len(config.LEARNERS), figsize=(14.0, 4.2), sharey=True)
    for idx, learner_name in enumerate(config.LEARNERS):
        ax = axes[idx]
        learner_subset = subset.loc[subset["learner_name"] == learner_name]
        theta_hat = learner_subset["theta_hat"].to_numpy(dtype=float)
        color = _LEARNER_COLORS[learner_name]

        ax.hist(theta_hat, bins=24, density=True, alpha=0.42, color=color)
        mean_hat = float(np.mean(theta_hat))
        sd_hat = float(np.std(theta_hat, ddof=1))

        if sd_hat > 0:
            x_min = float(np.min(theta_hat))
            x_max = float(np.max(theta_hat))
            x_grid = np.linspace(x_min, x_max, 300)
            normal_density = (
                1.0
                / (sd_hat * np.sqrt(2.0 * np.pi))
                * np.exp(-0.5 * ((x_grid - mean_hat) / sd_hat) ** 2)
            )
            ax.plot(x_grid, normal_density, color=color, linestyle="--", linewidth=1.7)

        ax.axvline(theta_true, color="black", linewidth=1.2)
        ax.axvline(mean_hat, color=color, linestyle=":", linewidth=1.3)
        ax.set_title(_LEARNER_LABELS[learner_name])
        ax.set_xlabel(r"$\hat{\theta}$")
        ax.grid(axis="y", alpha=0.25)

        if idx == 0:
            ax.set_ylabel("Density")

    legend_handles = [
        Patch(facecolor="#b3b3b3", alpha=0.42, label=r"Histogram of $\hat{\theta}$"),
        Line2D([0], [0], color="#555555", linestyle="--", linewidth=1.7, label="Normal approximation"),
        Line2D([0], [0], color="black", linewidth=1.2, label=r"True $\theta$"),
        Line2D([0], [0], color="#555555", linestyle=":", linewidth=1.3, label=r"Empirical mean of $\hat{\theta}$"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False)
    title = (
        "Sampling Distribution of DML Estimates: "
        f"{_DGP_LABELS[dgp_name]}, p={p_value}, n={requested_n_value}"
    )
    if actual_n_value != requested_n_value:
        title += f" (using available n={actual_n_value})"
    fig.suptitle(title, y=1.04)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_boxplot_thetahat(
    raw_df: pd.DataFrame,
    dgp_name: str,
    p_value: int,
    requested_n_value: int,
    output_path: Path,
) -> None:
    """Plot theta_hat boxplots by learner for one key scenario."""

    subset_all_n = raw_df.loc[(raw_df["dgp_name"] == dgp_name) & (raw_df["p"] == p_value)].copy()
    if subset_all_n.empty:
        raise ValueError(f"No replication data for dgp={dgp_name}, p={p_value}")

    available_n = sorted(int(value) for value in subset_all_n["n"].unique())
    if requested_n_value in available_n:
        actual_n_value = requested_n_value
    else:
        actual_n_value = min(available_n, key=lambda value: abs(value - requested_n_value))

    subset = subset_all_n.loc[subset_all_n["n"] == actual_n_value].copy()

    theta_true = float(subset["theta_true"].iloc[0])
    data = [
        subset.loc[subset["learner_name"] == learner_name, "theta_hat"].to_numpy(dtype=float)
        for learner_name in config.LEARNERS
    ]

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    box = ax.boxplot(
        data,
        labels=[_LEARNER_LABELS[name] for name in config.LEARNERS],
        patch_artist=True,
        widths=0.6,
        showfliers=False,
    )
    for patch, learner_name in zip(box["boxes"], config.LEARNERS, strict=False):
        patch.set_facecolor(_LEARNER_COLORS[learner_name])
        patch.set_alpha(0.45)
    ax.axhline(theta_true, color="black", linewidth=1.2, linestyle="--")
    title = f"Theta-Hat Boxplots: {_DGP_LABELS[dgp_name]}, p={p_value}, n={requested_n_value}"
    if actual_n_value != requested_n_value:
        title += f" (using available n={actual_n_value})"
    ax.set_title(title)
    ax.set_ylabel(r"$\hat{\theta}$")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_coverage_heatmap(
    metrics: pd.DataFrame, dgp_name: str, p_value: int, output_path: Path
) -> None:
    """Plot heatmap of empirical coverage over (n, learner) for one DGP x p cell."""

    subset = metrics.loc[(metrics["dgp_name"] == dgp_name) & (metrics["p"] == p_value)].copy()
    coverage_grid = (
        subset.pivot(index="n", columns="learner_name", values="coverage")
        .reindex(index=config.N_VALUES, columns=config.LEARNERS)
        .to_numpy(dtype=float)
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    image = ax.imshow(coverage_grid, aspect="auto", vmin=0.0, vmax=1.0, cmap="YlGnBu")
    ax.set_xticks(np.arange(len(config.LEARNERS)), [_LEARNER_LABELS[name] for name in config.LEARNERS])
    ax.set_yticks(np.arange(len(config.N_VALUES)), [str(value) for value in config.N_VALUES])
    ax.set_xlabel("Learner")
    ax.set_ylabel("Sample size n")
    ax.set_title(f"Coverage Heatmap: {_DGP_LABELS[dgp_name]}, p={p_value}")
    for row_idx, _ in enumerate(config.N_VALUES):
        for col_idx, _ in enumerate(config.LEARNERS):
            value = coverage_grid[row_idx, col_idx]
            ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", color="black")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Empirical 95% CI Coverage")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_legacy_coverage_grid(metrics: pd.DataFrame, output_path: Path) -> None:
    """Keep original aggregated coverage overview for backward compatibility."""

    ordered_dgps = list(config.DGP_NAMES)
    ordered_p = list(config.P_VALUES)
    ordered_n = list(config.N_VALUES)
    ordered_learners = list(config.LEARNERS)

    fig_cov, axes_cov = plt.subplots(
        nrows=len(ordered_dgps),
        ncols=len(ordered_p),
        figsize=(12, 7),
        sharex=True,
        sharey=True,
    )
    width = 0.22
    x_positions = list(range(len(ordered_n)))
    for dgp_index, dgp_name in enumerate(ordered_dgps):
        for p_index, p_value in enumerate(ordered_p):
            ax = axes_cov[dgp_index, p_index]
            for n_index, n_value in enumerate(ordered_n):
                learner_positions = _centered_learner_positions(x_center=n_index, width=width)
                for learner_offset, learner_name in enumerate(ordered_learners):
                    coverage_value = _cell_value(
                        metrics=metrics,
                        dgp_name=dgp_name,
                        p_value=p_value,
                        n_value=n_value,
                        learner_name=learner_name,
                        metric="coverage",
                    )
                    ax.bar(
                        learner_positions[learner_offset],
                        coverage_value,
                        width=width,
                        color=_LEARNER_COLORS[learner_name],
                        label=_LEARNER_LABELS[learner_name] if (n_index == 0) else None,
                    )
            ax.set_title(f"{dgp_name}, p={p_value}")
            ax.set_xticks(x_positions, [str(n) for n in ordered_n])
            if p_index == 0:
                ax.set_ylabel("Coverage")
            if dgp_index == len(ordered_dgps) - 1:
                ax.set_xlabel("n")
            ax.set_ylim(0.0, 1.0)
            ax.grid(axis="y", alpha=0.25)

    legend_handles = [
        Patch(facecolor=_LEARNER_COLORS[name], label=_LEARNER_LABELS[name]) for name in ordered_learners
    ]
    fig_cov.legend(handles=legend_handles, loc="upper center", ncol=len(ordered_learners))
    fig_cov.suptitle("Coverage by DGP, p, n, and Learner", y=0.98)
    fig_cov.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig_cov.savefig(output_path, dpi=200)
    plt.close(fig_cov)


def _plot_legacy_rmse_bias_grid(metrics: pd.DataFrame, output_path: Path) -> None:
    """Keep original aggregated RMSE/Bias overview for backward compatibility."""

    ordered_dgps = list(config.DGP_NAMES)
    ordered_p = list(config.P_VALUES)
    ordered_n = list(config.N_VALUES)
    ordered_learners = list(config.LEARNERS)
    x_positions = list(range(len(ordered_n)))

    fig_rb, axes_rb = plt.subplots(
        nrows=len(ordered_dgps),
        ncols=len(ordered_p),
        figsize=(12, 7),
        sharex=True,
    )
    learner_width = 0.24
    metric_width = 0.10
    for dgp_index, dgp_name in enumerate(ordered_dgps):
        for p_index, p_value in enumerate(ordered_p):
            ax = axes_rb[dgp_index, p_index]
            for n_index, n_value in enumerate(ordered_n):
                learner_positions = _centered_learner_positions(
                    x_center=n_index, width=learner_width
                )
                for learner_offset, learner_name in enumerate(ordered_learners):
                    rmse_value = _cell_value(
                        metrics=metrics,
                        dgp_name=dgp_name,
                        p_value=p_value,
                        n_value=n_value,
                        learner_name=learner_name,
                        metric="rmse",
                    )
                    bias_value = _cell_value(
                        metrics=metrics,
                        dgp_name=dgp_name,
                        p_value=p_value,
                        n_value=n_value,
                        learner_name=learner_name,
                        metric="bias",
                    )
                    center = learner_positions[learner_offset]
                    color = _LEARNER_COLORS[learner_name]
                    ax.bar(center - metric_width / 2, rmse_value, width=metric_width, color=color)
                    ax.bar(
                        center + metric_width / 2,
                        bias_value,
                        width=metric_width,
                        color=color,
                        hatch="//",
                        edgecolor="black",
                        linewidth=0.5,
                    )
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_title(f"{dgp_name}, p={p_value}")
            ax.set_xticks(x_positions, [str(n) for n in ordered_n])
            if p_index == 0:
                ax.set_ylabel("Metric Value")
            if dgp_index == len(ordered_dgps) - 1:
                ax.set_xlabel("n")
            ax.grid(axis="y", alpha=0.25)

    learner_handles = [
        Patch(facecolor=_LEARNER_COLORS[name], label=_LEARNER_LABELS[name]) for name in ordered_learners
    ]
    metric_handles = [
        Patch(facecolor="#777777", label="RMSE"),
        Patch(facecolor="#777777", hatch="//", edgecolor="black", label="Bias"),
    ]
    learner_legend = fig_rb.legend(
        handles=learner_handles, loc="upper center", ncol=len(ordered_learners)
    )
    fig_rb.add_artist(learner_legend)
    fig_rb.legend(handles=metric_handles, loc="upper right")
    fig_rb.suptitle("RMSE and Bias by DGP, p, n, and Learner", y=0.98)
    fig_rb.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig_rb.savefig(output_path, dpi=200)
    plt.close(fig_rb)


def task_figures(
    path_to_main_results: Path = (
        PROJECT_ROOT / f"documents/outputs/tables/main_results{SUFFIX}.csv"
    ),
    path_to_raw_results: Path = (
        PROJECT_ROOT / f"documents/outputs/raw/simulations{SUFFIX}.csv"
    ),
    path_to_figure_manifest: Annotated[Path, Product] = (
        PROJECT_ROOT / f"documents/outputs/figures/figure_manifest{SUFFIX}.txt"
    ),
) -> None:
    """Create figure suite for simulation diagnostics and finite-sample distributions."""

    expected_main_results = PROJECT_ROOT / f"documents/outputs/tables/main_results{SUFFIX}.csv"
    expected_raw_results = PROJECT_ROOT / f"documents/outputs/raw/simulations{SUFFIX}.csv"
    if path_to_main_results != expected_main_results:
        raise ValueError(
            f"task_figures must read {expected_main_results}, got {path_to_main_results}"
        )
    if path_to_raw_results != expected_raw_results:
        raise ValueError(f"task_figures must read {expected_raw_results}, got {path_to_raw_results}")

    results_df = pd.read_csv(path_to_main_results)
    raw_df = pd.read_csv(path_to_raw_results)
    metrics = _prepare_metrics_for_figures(results_df)
    figure_dir = PROJECT_ROOT / "documents/outputs/figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    generated_files: list[Path] = []
    for dgp_name in config.DGP_NAMES:
        dgp_stem = _DGP_FILE_STEMS[dgp_name]
        for p_value in config.P_VALUES:
            scenario_tag = f"{dgp_stem}_p{p_value}"

            coverage_path = figure_dir / f"coverage_{scenario_tag}{SUFFIX}.png"
            _plot_metric_by_n(
                metrics=metrics,
                dgp_name=dgp_name,
                p_value=p_value,
                metric_name="coverage",
                output_path=coverage_path,
                reference_line=0.95,
                y_bounds=(0.0, 1.0),
            )
            generated_files.append(coverage_path)

            rmse_path = figure_dir / f"rmse_{scenario_tag}{SUFFIX}.png"
            _plot_metric_by_n(
                metrics=metrics,
                dgp_name=dgp_name,
                p_value=p_value,
                metric_name="rmse",
                output_path=rmse_path,
            )
            generated_files.append(rmse_path)

            bias_path = figure_dir / f"bias_{scenario_tag}{SUFFIX}.png"
            _plot_metric_by_n(
                metrics=metrics,
                dgp_name=dgp_name,
                p_value=p_value,
                metric_name="bias",
                output_path=bias_path,
                zero_line=True,
            )
            generated_files.append(bias_path)

            abs_bias_path = figure_dir / f"abs_bias_{scenario_tag}{SUFFIX}.png"
            _plot_metric_by_n(
                metrics=metrics,
                dgp_name=dgp_name,
                p_value=p_value,
                metric_name="abs_bias",
                output_path=abs_bias_path,
            )
            generated_files.append(abs_bias_path)

            ci_length_path = figure_dir / f"ci_length_{scenario_tag}{SUFFIX}.png"
            _plot_metric_by_n(
                metrics=metrics,
                dgp_name=dgp_name,
                p_value=p_value,
                metric_name="mean_ci_length",
                output_path=ci_length_path,
            )
            generated_files.append(ci_length_path)

            heatmap_path = figure_dir / f"coverage_heatmap_{scenario_tag}{SUFFIX}.png"
            _plot_coverage_heatmap(
                metrics=metrics,
                dgp_name=dgp_name,
                p_value=p_value,
                output_path=heatmap_path,
            )
            generated_files.append(heatmap_path)

    key_distribution_scenarios = [
        ("linear_baseline", 100, 200),
        ("linear_baseline", 150, 300),
        ("linear_sparse_correlated", 100, 200),
        ("linear_sparse_correlated", 150, 300),
    ]
    for dgp_name, p_value, n_value in key_distribution_scenarios:
        dgp_stem = _DGP_FILE_STEMS[dgp_name]
        dist_path = figure_dir / f"dist_thetahat_{dgp_stem}_p{p_value}_n{n_value}{SUFFIX}.png"
        _plot_sampling_distribution(
            raw_df=raw_df,
            dgp_name=dgp_name,
            p_value=p_value,
            requested_n_value=n_value,
            output_path=dist_path,
        )
        generated_files.append(dist_path)

        boxplot_path = figure_dir / f"boxplot_thetahat_{dgp_stem}_p{p_value}_n{n_value}{SUFFIX}.png"
        _plot_boxplot_thetahat(
            raw_df=raw_df,
            dgp_name=dgp_name,
            p_value=p_value,
            requested_n_value=n_value,
            output_path=boxplot_path,
        )
        generated_files.append(boxplot_path)

    legacy_coverage_path = figure_dir / f"coverage{SUFFIX}.png"
    _plot_legacy_coverage_grid(metrics=metrics, output_path=legacy_coverage_path)
    generated_files.append(legacy_coverage_path)

    legacy_rmse_bias_path = figure_dir / f"rmse_bias{SUFFIX}.png"
    _plot_legacy_rmse_bias_grid(metrics=metrics, output_path=legacy_rmse_bias_path)
    generated_files.append(legacy_rmse_bias_path)

    path_to_figure_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_lines = [str(path.relative_to(PROJECT_ROOT)) for path in sorted(generated_files)]
    path_to_figure_manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
