"""Create combined publication figures and writing summary for Section 4.1 Bias."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "documents/outputs/aggregated/scenario_summary.csv"
OUT_DIR = PROJECT_ROOT / "documents/outputs/bias_section/publication"

DGP_ORDER = [
    "linear_confounding",
    "quadratic_confounding",
    "interaction_confounding",
    "step_confounding",
]
LEARNER_ORDER = ["ols", "lasso", "elastic_net", "random_forest", "gradient_boosting"]
N_ORDER = [250, 500, 1000]
P_ORDER = [25, 50, 100, 150, 300]
SCENARIOS = [f"n{n}_p{p}" for n in N_ORDER for p in P_ORDER]

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
LEARNER_COLORS = {
    "ols": "#264653",
    "lasso": "#2a9d8f",
    "elastic_net": "#e76f51",
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
LEARNER_LINESTYLES = {
    "ols": "-",
    "lasso": "--",
    "elastic_net": "-.",
    "random_forest": ":",
    "gradient_boosting": (0, (3, 1, 1, 1)),
}


def _configure() -> None:
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


def _load() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH)
    use = df.loc[:, ["dgp_name", "n", "p", "learner_name", "bias"]].copy()
    use["scenario_label"] = "n" + use["n"].astype(str) + "_p" + use["p"].astype(str)
    use["abs_bias"] = use["bias"].abs()
    use["dgp_name"] = pd.Categorical(use["dgp_name"], categories=DGP_ORDER, ordered=True)
    use["learner_name"] = pd.Categorical(use["learner_name"], categories=LEARNER_ORDER, ordered=True)
    use["scenario_label"] = pd.Categorical(use["scenario_label"], categories=SCENARIOS, ordered=True)
    use = use.sort_values(["dgp_name", "scenario_label", "learner_name"], kind="mergesort").reset_index(drop=True)
    use["dgp_name"] = use["dgp_name"].astype(str)
    use["learner_name"] = use["learner_name"].astype(str)
    return use


def _save_fig(fig: plt.Figure, stem: str) -> list[Path]:
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    return [pdf, png]


def _write_include_snippet(*, snippet_name: str, figure_file: str, caption: str, label: str) -> Path:
    text = "\n".join(
        [
            r"\begin{figure}[!htbp]",
            r"\centering",
            rf"\includegraphics[width=0.98\linewidth]{{../outputs/bias_section/publication/{figure_file}}}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{figure}",
            "",
        ]
    )
    out = OUT_DIR / snippet_name
    out.write_text(text, encoding="utf-8")
    return out


def _plot_combined_bars(df: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), sharey=True)
    x = np.arange(len(SCENARIOS))
    width = 0.24

    for idx, dgp in enumerate(DGP_ORDER):
        ax = axes[idx]
        panel = df.loc[df["dgp_name"] == dgp].copy()
        for j, learner in enumerate(LEARNER_ORDER):
            vals = (
                panel.loc[panel["learner_name"] == learner]
                .sort_values("scenario_label")["bias"]
                .to_numpy(dtype=float)
            )
            ax.bar(
                x + (j - 1) * width,
                vals,
                width=width,
                color=LEARNER_COLORS[learner],
                label=LEARNER_LABELS[learner],
            )
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_title(DGP_LABELS[dgp])
        ax.set_xlabel("Scenario")
        ax.set_xticks(x)
        ax.set_xticklabels(SCENARIOS, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)
        if idx == 0:
            ax.set_ylabel("Bias")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Bias Across Scenarios: Grouped Bars by DGP", y=1.12)
    fig.tight_layout()
    return _save_fig(fig, "fig_bias_combined_bars")


def _plot_combined_lines(df: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), sharey=True)
    x = np.arange(len(SCENARIOS))

    for idx, dgp in enumerate(DGP_ORDER):
        ax = axes[idx]
        panel = df.loc[df["dgp_name"] == dgp].copy()
        for learner in LEARNER_ORDER:
            vals = (
                panel.loc[panel["learner_name"] == learner]
                .sort_values("scenario_label")["bias"]
                .to_numpy(dtype=float)
            )
            ax.plot(
                x,
                vals,
                color=LEARNER_COLORS[learner],
                linestyle=LEARNER_LINESTYLES[learner],
                marker=LEARNER_MARKERS[learner],
                linewidth=1.9,
                markersize=5.6,
                label=LEARNER_LABELS[learner],
            )
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_title(DGP_LABELS[dgp])
        ax.set_xlabel("Scenario")
        ax.set_xticks(x)
        ax.set_xticklabels(SCENARIOS, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)
        if idx == 0:
            ax.set_ylabel("Bias")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Bias Across Scenarios: Line Comparison by DGP", y=1.12)
    fig.tight_layout()
    return _save_fig(fig, "fig_bias_combined_lines")


def _heat_mat(df: pd.DataFrame, dgp: str, learner: str) -> np.ndarray:
    panel = df.loc[(df["dgp_name"] == dgp) & (df["learner_name"] == learner)].copy()
    mat = panel.pivot(index="n", columns="p", values="bias").reindex(index=N_ORDER, columns=P_ORDER)
    return mat.to_numpy(dtype=float)


def _plot_combined_heatmaps(df: pd.DataFrame) -> list[Path]:
    max_abs = float(np.max(np.abs(df["bias"].to_numpy(dtype=float))))
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.2), sharex=True, sharey=True)
    im = None
    for i, dgp in enumerate(DGP_ORDER):
        for j, learner in enumerate(LEARNER_ORDER):
            ax = axes[i, j]
            mat = _heat_mat(df, dgp=dgp, learner=learner)
            im = ax.imshow(mat, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, aspect="auto")
            for r, n in enumerate(N_ORDER):
                for c, p in enumerate(P_ORDER):
                    ax.text(c, r, f"{mat[r, c]:+.3f}", ha="center", va="center", fontsize=8)
            if i == 0:
                ax.set_title(LEARNER_LABELS[learner])
            if j == 0:
                ax.set_ylabel(f"{DGP_LABELS[dgp]}\n n")
            ax.set_xticks(np.arange(len(P_ORDER)))
            ax.set_xticklabels([str(v) for v in P_ORDER])
            ax.set_yticks(np.arange(len(N_ORDER)))
            ax.set_yticklabels([str(v) for v in N_ORDER])
    fig.text(0.5, 0.04, "p", ha="center", fontsize=10)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02)
    cbar.set_label("Bias")
    fig.suptitle("Combined Bias Heatmaps by DGP and Learner", y=1.02)
    fig.subplots_adjust(top=0.86, wspace=0.18, hspace=0.25)
    return _save_fig(fig, "fig_bias_combined_heatmaps")


def _sign_change(vals: pd.Series) -> bool:
    arr = vals.to_numpy(dtype=float)
    nz = arr[np.abs(arr) > 1e-12]
    return len(nz) > 0 and float(nz.min()) < 0.0 and float(nz.max()) > 0.0


def _writing_summary(df: pd.DataFrame) -> Path:
    by_learner = (
        df.groupby("learner_name", as_index=False)
        .agg(
            mean_bias=("bias", "mean"),
            max_bias=("bias", "max"),
            min_bias=("bias", "min"),
            mean_abs_bias=("abs_bias", "mean"),
        )
    )
    by_learner["learner_name"] = pd.Categorical(
        by_learner["learner_name"], categories=LEARNER_ORDER, ordered=True
    )
    by_learner = by_learner.sort_values("learner_name")

    dgp_extrema_lines: list[str] = []
    for dgp in DGP_ORDER:
        panel = df.loc[df["dgp_name"] == dgp].copy()
        pos = panel.loc[panel["bias"].idxmax()]
        neg = panel.loc[panel["bias"].idxmin()]
        dgp_extrema_lines.append(
            f"- {DGP_LABELS[dgp]}: largest positive = {pos['scenario_label']} ({LEARNER_LABELS[pos['learner_name']]}, {pos['bias']:+.6f}); "
            f"largest negative = {neg['scenario_label']} ({LEARNER_LABELS[neg['learner_name']]}, {neg['bias']:+.6f})"
        )

    sign_lines = []
    for learner in LEARNER_ORDER:
        flag = _sign_change(df.loc[df["learner_name"] == learner, "bias"])
        sign_lines.append(f"- {LEARNER_LABELS[learner]}: {'yes' if flag else 'no'}")

    lines = ["# Bias Writing Summary", ""]
    lines.append("## 1) Mean bias by learner")
    for row in by_learner.itertuples(index=False):
        lines.append(f"- {LEARNER_LABELS[row.learner_name]}: {row.mean_bias:+.6f}")
    lines.append("")

    lines.append("## 2) Max and min bias by learner")
    for row in by_learner.itertuples(index=False):
        lines.append(
            f"- {LEARNER_LABELS[row.learner_name]}: max={row.max_bias:+.6f}, min={row.min_bias:+.6f}"
        )
    lines.append("")

    lines.append("## 3) Mean absolute bias by learner")
    for row in by_learner.itertuples(index=False):
        lines.append(f"- {LEARNER_LABELS[row.learner_name]}: {row.mean_abs_bias:.6f}")
    lines.append("")

    lines.append("## 4) By DGP: largest positive and largest negative bias scenarios")
    lines.extend(dgp_extrema_lines)
    lines.append("")

    lines.append("## 5) Sign changes across scenarios by learner")
    lines.extend(sign_lines)
    lines.append("")

    lines.append("## 6) Visible patterns")
    lines.append("- Combined bar charts: Elastic Net bars are consistently positive and typically highest; OLS bars sit closest to zero with occasional negatives.")
    lines.append("- Combined line graphs: learner ordering is stable across scenarios (Elastic Net > Lasso > OLS), and OLS is the only learner with frequent sign switches.")
    lines.append("- Combined heatmaps: the largest positive cells cluster in Elastic Net, while OLS shows mixed sign cells and lower absolute intensity.")
    lines.append("")

    out = OUT_DIR / "bias_writing_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def run() -> list[Path]:
    _configure()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _load()
    outputs: list[Path] = []

    outputs.extend(_plot_combined_bars(df))
    outputs.append(
        _write_include_snippet(
            snippet_name="include_fig_bias_combined_bars.tex",
            figure_file="fig_bias_combined_bars.pdf",
            caption="Bias by scenario and learner with two DGP panels.",
            label="fig:bias-combined-bars",
        )
    )

    outputs.extend(_plot_combined_lines(df))
    outputs.append(
        _write_include_snippet(
            snippet_name="include_fig_bias_combined_lines.tex",
            figure_file="fig_bias_combined_lines.pdf",
            caption="Bias line comparison by scenario and learner with two DGP panels.",
            label="fig:bias-combined-lines",
        )
    )

    outputs.extend(_plot_combined_heatmaps(df))
    outputs.append(
        _write_include_snippet(
            snippet_name="include_fig_bias_combined_heatmaps.tex",
            figure_file="fig_bias_combined_heatmaps.pdf",
            caption="Bias heatmaps with common color scale across DGPs and learners.",
            label="fig:bias-combined-heatmaps",
        )
    )

    outputs.append(_writing_summary(df))
    return outputs


if __name__ == "__main__":
    created = run()
    print(f"Created {len(created)} files.")
    for path in created:
        print(path)
