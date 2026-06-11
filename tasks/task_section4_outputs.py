"""Generate Section 4 tables and figures (RMSE, coverage, t-stat diagnostics)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "documents/outputs/tables"
FIGURE_DIR = PROJECT_ROOT / "documents/outputs/figures/paper"
RMSE_DIR = PROJECT_ROOT / "documents/outputs/rmse_section"
COVERAGE_DIR = PROJECT_ROOT / "documents/outputs/coverage_section"
TSTAT_DIR = PROJECT_ROOT / "documents/outputs/tstat_section"

MAIN_RESULTS_PATH = TABLE_DIR / "table_main_results.csv"
SCENARIO_SUMMARY_PATH = PROJECT_ROOT / "documents/outputs/aggregated/scenario_summary.csv"

DGP_ORDER = [
    "linear_confounding",
    "quadratic_confounding",
    "interaction_confounding",
    "step_confounding",
]
DGP_LABELS = {
    "linear_confounding": "Linear",
    "quadratic_confounding": "Quadratic",
    "interaction_confounding": "Interaction",
    "step_confounding": "Step",
}
LEARNER_ORDER = ["ols", "lasso", "elastic_net", "random_forest", "gradient_boosting"]
LEARNER_LABELS = {
    "ols": "OLS",
    "lasso": "Lasso",
    "elastic_net": "Elastic Net",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}
SCENARIO_ORDER = [
    f"n{n}_p{p}"
    for n in [250, 500, 1000]
    for p in [25, 50, 100, 150, 300]
]

COLORS = {
    "ols": "#264653",
    "lasso": "#2a9d8f",
    "elastic_net": "#e76f51",
    "random_forest": "#b279a2",
    "gradient_boosting": "#e45756",
}
MARKERS = {
    "ols": "o",
    "lasso": "s",
    "elastic_net": "^",
    "random_forest": "D",
    "gradient_boosting": "v",
}
LINESTYLES = {
    "ols": "-",
    "lasso": "--",
    "elastic_net": "-.",
    "random_forest": ":",
    "gradient_boosting": (0, (3, 1, 1, 1)),
}


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 220,
            "savefig.bbox": "tight",
        }
    )


def _to_markdown(df: pd.DataFrame, caption: str) -> str:
    headers = [str(c) for c in df.columns]
    lines = [f"**{caption}**", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in df.itertuples(index=False, name=None):
        vals = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                vals.append(f"{float(v):.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    return "\n".join(lines)


def _load_main_results() -> pd.DataFrame:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    if MAIN_RESULTS_PATH.exists():
        df = pd.read_csv(MAIN_RESULTS_PATH)
    else:
        df = pd.read_csv(SCENARIO_SUMMARY_PATH)
        df.to_csv(MAIN_RESULTS_PATH, index=False, encoding="utf-8")

    required = {
        "dgp_name",
        "n",
        "p",
        "learner_name",
        "bias",
        "rmse",
        "coverage",
        "t_stat_mean",
        "t_stat_sd",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"main_results missing required columns: {missing}")

    out = df.copy()
    out["scenario"] = "n" + out["n"].astype(int).astype(str) + "_p" + out["p"].astype(int).astype(str)
    out["dgp_name"] = pd.Categorical(out["dgp_name"], categories=DGP_ORDER, ordered=True)
    out["learner_name"] = pd.Categorical(out["learner_name"], categories=LEARNER_ORDER, ordered=True)
    out["scenario"] = pd.Categorical(out["scenario"], categories=SCENARIO_ORDER, ordered=True)
    out = out.sort_values(["dgp_name", "scenario", "learner_name"], kind="mergesort").reset_index(drop=True)
    out["dgp_name"] = out["dgp_name"].astype(str)
    out["learner_name"] = out["learner_name"].astype(str)
    return out


def _wide_metric_table(df: pd.DataFrame, metric: str, metric_title: str) -> pd.DataFrame:
    base = df.loc[:, ["dgp_name", "scenario", "n", "p", "learner_name", metric]].copy()
    base["dgp"] = base["dgp_name"].map(DGP_LABELS)
    base["learner"] = base["learner_name"].map(LEARNER_LABELS)
    wide = (
        base.pivot_table(
            index=["dgp", "scenario", "n", "p"],
            columns="learner",
            values=metric,
            aggfunc="first",
        )
        .reindex(columns=["OLS", "Lasso", "Elastic Net"])
        .reset_index()
        .rename(columns={"scenario": "scenario_label"})
    )
    wide = wide.sort_values(["dgp", "n", "p"], kind="mergesort").reset_index(drop=True)
    wide = wide.rename(columns={"dgp": "DGP", "scenario_label": "scenario"})
    numeric_cols = ["OLS", "Lasso", "Elastic Net"]
    wide[numeric_cols] = wide[numeric_cols].round(4)
    return wide


def _plot_two_panel_grouped_bar(
    df: pd.DataFrame,
    metric: str,
    y_label: str,
    title: str,
    out_name: str,
    *,
    reference_line: float | None = None,
    y_lim: tuple[float, float] | None = None,
) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), sharey=True)
    x = np.arange(len(SCENARIO_ORDER))
    width = 0.24

    for i, dgp in enumerate(DGP_ORDER):
        ax = axes[i]
        panel = df.loc[df["dgp_name"] == dgp].copy()
        for j, learner in enumerate(LEARNER_ORDER):
            vals = (
                panel.loc[panel["learner_name"] == learner]
                .sort_values("scenario")[metric]
                .to_numpy(dtype=float)
            )
            ax.bar(
                x + (j - 1) * width,
                vals,
                width=width,
                color=COLORS[learner],
                label=LEARNER_LABELS[learner],
            )
        if reference_line is not None:
            ax.axhline(reference_line, color="black", linewidth=1.0, linestyle="--")
        ax.set_title(DGP_LABELS[dgp])
        ax.set_xlabel("Scenario")
        ax.set_xticks(x)
        ax.set_xticklabels(SCENARIO_ORDER, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)
        if i == 0:
            ax.set_ylabel(y_label)
        if y_lim is not None:
            ax.set_ylim(*y_lim)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(title, y=1.12)
    fig.tight_layout()

    out_path = FIGURE_DIR / out_name
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def _plot_two_panel_lines(
    df: pd.DataFrame,
    metric: str,
    y_label: str,
    title: str,
    out_name: str,
    *,
    reference_line: float | None = None,
    y_lim: tuple[float, float] | None = None,
) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), sharey=True)
    x = np.arange(len(SCENARIO_ORDER))

    for i, dgp in enumerate(DGP_ORDER):
        ax = axes[i]
        panel = df.loc[df["dgp_name"] == dgp].copy()
        for learner in LEARNER_ORDER:
            vals = (
                panel.loc[panel["learner_name"] == learner]
                .sort_values("scenario")[metric]
                .to_numpy(dtype=float)
            )
            ax.plot(
                x,
                vals,
                color=COLORS[learner],
                marker=MARKERS[learner],
                linestyle=LINESTYLES[learner],
                linewidth=1.9,
                markersize=5.6,
                label=LEARNER_LABELS[learner],
            )
        if reference_line is not None:
            ax.axhline(reference_line, color="black", linewidth=1.0, linestyle="--")
        ax.set_title(DGP_LABELS[dgp])
        ax.set_xlabel("Scenario")
        ax.set_xticks(x)
        ax.set_xticklabels(SCENARIO_ORDER, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)
        if i == 0:
            ax.set_ylabel(y_label)
        if y_lim is not None:
            ax.set_ylim(*y_lim)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(title, y=1.12)
    fig.tight_layout()

    out_path = FIGURE_DIR / out_name
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def _write_rmse_outputs(df: pd.DataFrame) -> list[Path]:
    out: list[Path] = []
    table = _wide_metric_table(df, metric="rmse", metric_title="RMSE")
    rmse_tables = RMSE_DIR / "tables"
    rmse_figs = RMSE_DIR / "figures"
    rmse_tables.mkdir(parents=True, exist_ok=True)
    rmse_figs.mkdir(parents=True, exist_ok=True)
    csv = rmse_tables / "rmse_table_by_dgp.csv"
    md = rmse_tables / "rmse_table_by_dgp.md"
    table.to_csv(csv, index=False, encoding="utf-8")
    md.write_text(_to_markdown(table, "RMSE by DGP and simulation scenario."), encoding="utf-8")
    out.extend([csv, md])

    fig1 = _plot_two_panel_grouped_bar(
        df,
        metric="rmse",
        y_label="RMSE",
        title="RMSE Across Scenarios by DGP",
        out_name="fig_rmse_grouped_bar.png",
    )
    fig2 = _plot_two_panel_lines(
        df,
        metric="rmse",
        y_label="RMSE",
        title="RMSE Comparison Across Scenarios by DGP",
        out_name="fig_rmse_line.png",
    )
    target1 = rmse_figs / fig1.name
    target2 = rmse_figs / fig2.name
    Path(fig1).replace(target1)
    Path(fig2).replace(target2)
    out.extend([target1, target2])
    return out


def _write_coverage_outputs(df: pd.DataFrame) -> list[Path]:
    out: list[Path] = []
    table = _wide_metric_table(df, metric="coverage", metric_title="Coverage")
    cov_tables = COVERAGE_DIR / "tables"
    cov_figs = COVERAGE_DIR / "figures"
    cov_tables.mkdir(parents=True, exist_ok=True)
    cov_figs.mkdir(parents=True, exist_ok=True)
    csv = cov_tables / "table_coverage_by_dgp.csv"
    md = cov_tables / "table_coverage_by_dgp.md"
    table.to_csv(csv, index=False, encoding="utf-8")
    md.write_text(
        _to_markdown(table, "Confidence interval coverage by DGP and simulation scenario."),
        encoding="utf-8",
    )
    out.extend([csv, md])

    cov_vals = df["coverage"].to_numpy(dtype=float)
    y_min = max(0.0, float(np.min(cov_vals)) - 0.03)
    y_max = min(1.0, float(np.max(cov_vals)) + 0.03)
    y_lim = (y_min, y_max)

    fig1 = _plot_two_panel_grouped_bar(
        df,
        metric="coverage",
        y_label="Coverage",
        title="Coverage Across Scenarios by DGP",
        out_name="fig_coverage_grouped_bar.png",
        reference_line=0.95,
        y_lim=y_lim,
    )
    fig2 = _plot_two_panel_lines(
        df,
        metric="coverage",
        y_label="Coverage",
        title="Coverage Comparison Across Scenarios by DGP",
        out_name="fig_coverage_line.png",
        reference_line=0.95,
        y_lim=y_lim,
    )
    target1 = cov_figs / fig1.name
    target2 = cov_figs / fig2.name
    Path(fig1).replace(target1)
    Path(fig2).replace(target2)
    out.extend([target1, target2])
    return out


def _write_tstat_outputs(df: pd.DataFrame) -> list[Path]:
    out: list[Path] = []
    t_tables = TSTAT_DIR / "tables"
    t_figs = TSTAT_DIR / "figures"
    t_tables.mkdir(parents=True, exist_ok=True)
    t_figs.mkdir(parents=True, exist_ok=True)
    base = df.loc[:, ["dgp_name", "n", "p", "scenario", "learner_name", "t_stat_mean", "t_stat_sd"]].copy()
    base["DGP"] = base["dgp_name"].map(DGP_LABELS)
    base["learner"] = base["learner_name"].map(LEARNER_LABELS)

    wide = (
        base.pivot_table(
            index=["DGP", "scenario", "n", "p"],
            columns="learner",
            values=["t_stat_mean", "t_stat_sd"],
            aggfunc="first",
        )
        .reindex(
            columns=pd.MultiIndex.from_tuples(
                [
                    ("t_stat_mean", "OLS"),
                    ("t_stat_sd", "OLS"),
                    ("t_stat_mean", "Lasso"),
                    ("t_stat_sd", "Lasso"),
                    ("t_stat_mean", "Elastic Net"),
                    ("t_stat_sd", "Elastic Net"),
                ]
            )
        )
        .reset_index()
    )
    wide.columns = [
        "DGP",
        "scenario",
        "n",
        "p",
        "OLS_mean_t",
        "OLS_sd_t",
        "Lasso_mean_t",
        "Lasso_sd_t",
        "ElasticNet_mean_t",
        "ElasticNet_sd_t",
    ]
    wide = wide.sort_values(["DGP", "n", "p"], kind="mergesort").reset_index(drop=True)
    numeric_cols = [
        "OLS_mean_t",
        "OLS_sd_t",
        "Lasso_mean_t",
        "Lasso_sd_t",
        "ElasticNet_mean_t",
        "ElasticNet_sd_t",
    ]
    wide[numeric_cols] = wide[numeric_cols].round(4)

    csv = t_tables / "tstat_diagnostics_table.csv"
    md = t_tables / "tstat_diagnostics_table.md"
    wide.to_csv(csv, index=False, encoding="utf-8")
    md.write_text(
        _to_markdown(
            wide,
            "Monte Carlo mean and standard deviation of t-statistics by DGP and scenario.",
        ),
        encoding="utf-8",
    )
    out.extend([csv, md])

    fig1 = _plot_two_panel_grouped_bar(
        df,
        metric="t_stat_sd",
        y_label="t-statistic SD",
        title="Standard Deviation of t-Statistics Across Scenarios",
        out_name="fig_tstat_sd_grouped_bar.png",
        reference_line=1.0,
    )
    fig2 = _plot_two_panel_lines(
        df,
        metric="t_stat_mean",
        y_label="Mean t-statistic",
        title="Mean t-Statistic Across Scenarios",
        out_name="fig_tstat_mean_line.png",
        reference_line=0.0,
    )
    target1 = t_figs / fig1.name
    target2 = t_figs / fig2.name
    Path(fig1).replace(target1)
    Path(fig2).replace(target2)
    out.extend([target1, target2])
    return out


def _write_section4_summary_table(df: pd.DataFrame) -> list[Path]:
    out: list[Path] = []
    overall = (
        df.groupby("learner_name", as_index=False)
        .agg(
            mean_bias=("bias", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_coverage=("coverage", "mean"),
            mean_t_stat_mean=("t_stat_mean", "mean"),
            mean_t_stat_sd=("t_stat_sd", "mean"),
        )
        .copy()
    )
    overall["learner_name"] = pd.Categorical(overall["learner_name"], categories=LEARNER_ORDER, ordered=True)
    overall = overall.sort_values("learner_name").reset_index(drop=True)

    by_dgp = (
        df.groupby(["learner_name", "dgp_name"], as_index=False)
        .agg(
            mean_bias=("bias", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_coverage=("coverage", "mean"),
            mean_t_stat_mean=("t_stat_mean", "mean"),
            mean_t_stat_sd=("t_stat_sd", "mean"),
        )
        .copy()
    )

    summary = overall.copy()
    for dgp in DGP_ORDER:
        part = by_dgp.loc[by_dgp["dgp_name"] == dgp, :].copy()
        part["learner_name"] = pd.Categorical(part["learner_name"], categories=LEARNER_ORDER, ordered=True)
        part = part.sort_values("learner_name")
        suffix = dgp
        summary[f"mean_bias_{suffix}"] = part["mean_bias"].to_numpy()
        summary[f"mean_rmse_{suffix}"] = part["mean_rmse"].to_numpy()
        summary[f"mean_coverage_{suffix}"] = part["mean_coverage"].to_numpy()
        summary[f"mean_t_stat_mean_{suffix}"] = part["mean_t_stat_mean"].to_numpy()
        summary[f"mean_t_stat_sd_{suffix}"] = part["mean_t_stat_sd"].to_numpy()

    summary = summary.rename(columns={"learner_name": "learner"})
    summary["learner"] = summary["learner"].map(LEARNER_LABELS)

    numeric_cols = [c for c in summary.columns if c != "learner"]
    summary[numeric_cols] = summary[numeric_cols].round(4)

    csv = TABLE_DIR / "section4_summary_table.csv"
    md = TABLE_DIR / "section4_summary_table.md"
    summary.to_csv(csv, index=False, encoding="utf-8")
    md.write_text(_to_markdown(summary, "Average finite-sample performance by learner."), encoding="utf-8")
    out.extend([csv, md])
    return out


def run() -> list[Path]:
    _configure_style()
    df = _load_main_results()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = [MAIN_RESULTS_PATH]
    outputs.extend(_write_rmse_outputs(df))
    outputs.extend(_write_coverage_outputs(df))
    outputs.extend(_write_tstat_outputs(df))
    outputs.extend(_write_section4_summary_table(df))
    return outputs


if __name__ == "__main__":
    files = run()
    print(f"Created {len(files)} files:")
    for file in files:
        print(file)
