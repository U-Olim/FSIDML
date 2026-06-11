"""Generate Section 4.4 diagnostic figures for t-statistic calibration."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "documents/outputs/raw/simulations.csv"
TSTAT_TABLE_PATH = PROJECT_ROOT / "documents/outputs/t-stat_section/tstat_diagnostics_table.csv"
OUT_DIR = PROJECT_ROOT / "documents/outputs/t-stat_section"

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
LEARNER_COLORS = {
    "ols": "#264653",
    "lasso": "#2a9d8f",
    "elastic_net": "#e76f51",
    "random_forest": "#b279a2",
    "gradient_boosting": "#e45756",
}
SCENARIO_ORDER = [
    f"n{n}_p{p}"
    for n in [250, 500, 1000]
    for p in [25, 50, 100, 150, 300]
]


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
            "figure.dpi": 240,
            "savefig.bbox": "tight",
        }
    )


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(RAW_PATH)
    raw["t_stat"] = (raw["theta_hat"] - raw["theta_true"]) / raw["se"]
    raw["scenario"] = "n" + raw["n"].astype(int).astype(str) + "_p" + raw["p"].astype(int).astype(str)
    raw["dgp_name"] = pd.Categorical(raw["dgp_name"], categories=DGP_ORDER, ordered=True)
    raw["learner_name"] = pd.Categorical(raw["learner_name"], categories=LEARNER_ORDER, ordered=True)
    raw["scenario"] = pd.Categorical(raw["scenario"], categories=SCENARIO_ORDER, ordered=True)
    raw = raw.sort_values(["dgp_name", "scenario", "learner_name"], kind="mergesort").reset_index(drop=True)
    raw["dgp_name"] = raw["dgp_name"].astype(str)
    raw["learner_name"] = raw["learner_name"].astype(str)

    if TSTAT_TABLE_PATH.exists():
        tbl = pd.read_csv(TSTAT_TABLE_PATH)
        # Map labels back to canonical keys for robust ordering.
        dgp_inv = {v: k for k, v in DGP_LABELS.items()}
        learner_inv = {"OLS": "ols", "Lasso": "lasso", "ElasticNet": "elastic_net"}
        tbl["dgp_name"] = tbl["DGP"].map(dgp_inv)
        tbl["learner_name"] = tbl["Learner"].map(learner_inv)
        tbl["scenario"] = "n" + tbl["n"].astype(int).astype(str) + "_p" + tbl["p"].astype(int).astype(str)
        tbl = tbl.rename(columns={"Mean(t)": "mean_t", "SD(t)": "sd_t"})
        tbl["dgp_name"] = pd.Categorical(tbl["dgp_name"], categories=DGP_ORDER, ordered=True)
        tbl["learner_name"] = pd.Categorical(tbl["learner_name"], categories=LEARNER_ORDER, ordered=True)
        tbl["scenario"] = pd.Categorical(tbl["scenario"], categories=SCENARIO_ORDER, ordered=True)
        tbl = tbl.sort_values(["dgp_name", "scenario", "learner_name"], kind="mergesort").reset_index(drop=True)
        tbl["dgp_name"] = tbl["dgp_name"].astype(str)
        tbl["learner_name"] = tbl["learner_name"].astype(str)
        return raw, tbl

    agg = (
        raw.groupby(["dgp_name", "scenario", "n", "p", "learner_name"], as_index=False)
        .agg(
            mean_t=("t_stat", "mean"),
            sd_t=("t_stat", lambda s: float(np.std(s, ddof=1))),
        )
    )
    return raw, agg


def _kde_gaussian(samples: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    n = samples.size
    if n < 2:
        return np.zeros_like(x_grid)
    std = float(np.std(samples, ddof=1))
    if std <= 0:
        return np.zeros_like(x_grid)
    h = 1.06 * std * (n ** (-1.0 / 5.0))
    if h <= 0:
        return np.zeros_like(x_grid)
    u = (x_grid[:, None] - samples[None, :]) / h
    return np.mean(np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi), axis=1) / h


def _save(fig: plt.Figure, stem: str) -> list[Path]:
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def _plot_grouped_bar(
    summary: pd.DataFrame,
    *,
    metric: str,
    y_label: str,
    title: str,
    ref_line: float,
    out_stem: str,
) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), sharey=True)
    x = np.arange(len(SCENARIO_ORDER))
    width = 0.24

    for i, dgp in enumerate(DGP_ORDER):
        ax = axes[i]
        panel = summary.loc[summary["dgp_name"] == dgp]
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
                color=LEARNER_COLORS[learner],
                label=LEARNER_LABELS[learner],
            )
        ax.axhline(ref_line, color="black", linewidth=1.0, linestyle="--")
        ax.set_title(DGP_LABELS[dgp])
        ax.set_xlabel("Scenario")
        ax.set_xticks(x)
        ax.set_xticklabels(SCENARIO_ORDER, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)
        if i == 0:
            ax.set_ylabel(y_label)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.suptitle(title, y=1.12)
    fig.tight_layout()
    return _save(fig, out_stem)


def _plot_distribution_for_scenario(raw: pd.DataFrame, *, n: int, p: int, out_stem: str) -> list[Path]:
    sub = raw.loc[(raw["n"] == n) & (raw["p"] == p)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)

    # Common x-range for this scenario across both DGP panels.
    x_min = float(np.quantile(sub["t_stat"].to_numpy(dtype=float), 0.005))
    x_max = float(np.quantile(sub["t_stat"].to_numpy(dtype=float), 0.995))
    x_min = min(x_min, -4.0)
    x_max = max(x_max, 4.0)
    bound = max(abs(x_min), abs(x_max))
    x_grid = np.linspace(-bound, bound, 900)
    normal = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x_grid * x_grid)

    for i, dgp in enumerate(DGP_ORDER):
        ax = axes[i]
        panel = sub.loc[sub["dgp_name"] == dgp]
        for learner in LEARNER_ORDER:
            vals = panel.loc[panel["learner_name"] == learner, "t_stat"].to_numpy(dtype=float)
            dens = _kde_gaussian(vals, x_grid=x_grid)
            ax.plot(x_grid, dens, color=LEARNER_COLORS[learner], linewidth=2.0, label=LEARNER_LABELS[learner])
        ax.plot(x_grid, normal, color="black", linestyle="--", linewidth=1.4, label="N(0,1)")
        ax.set_title(DGP_LABELS[dgp])
        ax.set_xlabel("t-statistic")
        ax.grid(axis="y", alpha=0.25)
        if i == 0:
            ax.set_ylabel("Density")
        ax.set_xlim(-bound, bound)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.08), frameon=False)
    fig.suptitle(f"t-Statistic Distribution, n={n}, p={p}", y=1.13)
    fig.tight_layout()
    return _save(fig, out_stem)


def _plot_sd_heatmap(summary: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.2), sharey=True)
    vmax = float(summary["sd_t"].max())
    vmin = float(summary["sd_t"].min())

    for i, dgp in enumerate(DGP_ORDER):
        ax = axes[i]
        panel = summary.loc[summary["dgp_name"] == dgp].copy()
        matrix = (
            panel.pivot_table(index="scenario", columns="learner_name", values="sd_t", aggfunc="first")
            .reindex(index=SCENARIO_ORDER, columns=LEARNER_ORDER)
            .to_numpy(dtype=float)
        )
        im = ax.imshow(matrix, cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="auto")
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                ax.text(c, r, f"{matrix[r, c]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(DGP_LABELS[dgp])
        ax.set_xticks(np.arange(len(LEARNER_ORDER)))
        ax.set_xticklabels([LEARNER_LABELS[x] for x in LEARNER_ORDER], rotation=15, ha="right")
        ax.set_yticks(np.arange(len(SCENARIO_ORDER)))
        ax.set_yticklabels(SCENARIO_ORDER)
        if i == 0:
            ax.set_ylabel("Scenario")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9, pad=0.02)
    cbar.set_label("SD(t)")
    fig.suptitle("SD(t) Heatmap Across Scenarios and Learners", y=1.03)
    fig.subplots_adjust(top=0.86, wspace=0.20)
    return _save(fig, "fig_tstat_sd_heatmap")


def run() -> list[Path]:
    _configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw, summary = _load_data()

    outputs: list[Path] = []
    outputs.extend(
        _plot_grouped_bar(
            summary,
            metric="sd_t",
            y_label="SD(t)",
            title="Standard Deviation of t-Statistics Across Scenarios",
            ref_line=1.0,
            out_stem="fig_tstat_sd_grouped_bar",
        )
    )
    outputs.extend(
        _plot_grouped_bar(
            summary,
            metric="mean_t",
            y_label="Mean(t)",
            title="Mean t-Statistics Across Scenarios",
            ref_line=0.0,
            out_stem="fig_tstat_mean_grouped_bar",
        )
    )
    outputs.extend(
        _plot_distribution_for_scenario(
            raw,
            n=200,
            p=100,
            out_stem="fig_tstat_distribution_n200_p100",
        )
    )
    outputs.extend(
        _plot_distribution_for_scenario(
            raw,
            n=300,
            p=150,
            out_stem="fig_tstat_distribution_n300_p150",
        )
    )
    outputs.extend(_plot_sd_heatmap(summary))
    return outputs


if __name__ == "__main__":
    created = run()
    print("Created files:")
    for path in created:
        print(path)
    print("\nMain-text recommendation:")
    print("- fig_tstat_sd_grouped_bar.png")
    print("- fig_tstat_distribution_n200_p100.png")
    print("\nAppendix/supplement recommendation:")
    print("- fig_tstat_mean_grouped_bar.png")
    print("- fig_tstat_distribution_n300_p150.png")
    print("- fig_tstat_sd_heatmap.png")
