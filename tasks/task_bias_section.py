"""Generate Section 4.1 (Bias) diagnostics and publication-ready artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "documents/outputs/aggregated/scenario_summary.csv"
OUTPUT_ROOT = PROJECT_ROOT / "documents/outputs/bias_section"

DGP_ORDER = ["linear_baseline", "linear_sparse_correlated"]
LEARNER_ORDER = ["ols", "lasso", "elastic_net"]
N_ORDER = [200, 300, 400]
P_ORDER = [100, 150]

DGP_LABELS = {
    "linear_baseline": "Linear Baseline",
    "linear_sparse_correlated": "Linear Sparse Correlated",
}
LEARNER_LABELS = {"ols": "OLS", "lasso": "Lasso", "elastic_net": "Elastic Net"}
LEARNER_COLORS = {"ols": "#264653", "lasso": "#2a9d8f", "elastic_net": "#e76f51"}
LEARNER_MARKERS = {"ols": "o", "lasso": "s", "elastic_net": "^"}
LEARNER_LINESTYLES = {"ols": "-", "lasso": "--", "elastic_net": "-."}


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _df_to_latex(
    df: pd.DataFrame,
    *,
    float_decimals: int,
    caption: str | None = None,
    label: str | None = None,
    longtable: bool = False,
) -> str:
    cols = list(df.columns)
    align = "l" + "r" * (len(cols) - 1)
    lines: list[str] = []
    if longtable:
        # Keep output LaTeX-simple and paper-ready under a standard floating table.
        longtable = False
    lines.append("\\begin{table}[!htbp]")
    lines.append("\\centering")
    if caption:
        lines.append(f"\\caption{{{_latex_escape(caption)}}}")
    if label:
        lines.append(f"\\label{{{_latex_escape(label)}}}")
    lines.append(f"\\begin{{tabular}}{{{align}}}")
    lines.append(" \\hline")
    header = " & ".join(_latex_escape(col) for col in cols) + r" \\"
    lines.append(header)
    lines.append(" \\hline")
    for row in df.itertuples(index=False, name=None):
        rendered: list[str] = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                rendered.append(f"{float(value):.{float_decimals}f}")
            else:
                rendered.append(_latex_escape(value))
        lines.append(" & ".join(rendered) + r" \\")
    lines.append(" \\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def _load_bias_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"dgp_name", "n", "p", "learner_name", "bias"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"scenario_summary missing required columns: {sorted(missing)}")

    out = df.loc[:, ["dgp_name", "n", "p", "learner_name", "bias"]].copy()
    out["abs_bias"] = out["bias"].abs()
    out["scenario_label"] = "n" + out["n"].astype(str) + "_p" + out["p"].astype(str)

    out["dgp_name"] = pd.Categorical(out["dgp_name"], categories=DGP_ORDER, ordered=True)
    out["learner_name"] = pd.Categorical(out["learner_name"], categories=LEARNER_ORDER, ordered=True)
    out["n"] = pd.Categorical(out["n"], categories=N_ORDER, ordered=True)
    out["p"] = pd.Categorical(out["p"], categories=P_ORDER, ordered=True)
    out = out.sort_values(["dgp_name", "n", "p", "learner_name"], kind="mergesort").reset_index(drop=True)
    out["dgp_name"] = out["dgp_name"].astype(str)
    out["learner_name"] = out["learner_name"].astype(str)
    out["n"] = out["n"].astype(int)
    out["p"] = out["p"].astype(int)
    return out


def _scenario_order_labels() -> list[str]:
    return [f"n{n}_p{p}" for n in N_ORDER for p in P_ORDER]


def _write_pivot_tables(df: pd.DataFrame, diagnostics_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    index_cols = ["dgp_name", "n", "p"]
    bias_pivot = (
        df.pivot_table(index=index_cols, columns="learner_name", values="bias", aggfunc="first")
        .reindex(columns=LEARNER_ORDER)
        .reset_index()
    )
    abs_bias_pivot = (
        df.pivot_table(index=index_cols, columns="learner_name", values="abs_bias", aggfunc="first")
        .reindex(columns=LEARNER_ORDER)
        .reset_index()
    )

    bias_csv = diagnostics_dir / "bias_pivot_by_dgp_n_p_x_learner.csv"
    abs_bias_csv = diagnostics_dir / "abs_bias_pivot_by_dgp_n_p_x_learner.csv"
    bias_tex = diagnostics_dir / "bias_pivot_by_dgp_n_p_x_learner.tex"
    abs_bias_tex = diagnostics_dir / "abs_bias_pivot_by_dgp_n_p_x_learner.tex"

    bias_pivot.to_csv(bias_csv, index=False, encoding="utf-8")
    abs_bias_pivot.to_csv(abs_bias_csv, index=False, encoding="utf-8")

    bias_tex_text = _df_to_latex(bias_pivot, float_decimals=6)
    abs_bias_tex_text = _df_to_latex(abs_bias_pivot, float_decimals=6)
    bias_tex.write_text(bias_tex_text, encoding="utf-8")
    abs_bias_tex.write_text(abs_bias_tex_text, encoding="utf-8")

    return bias_pivot, abs_bias_pivot


def _heatmap_matrix(df: pd.DataFrame, dgp_name: str, learner_name: str, metric: str) -> np.ndarray:
    subset = df.loc[(df["dgp_name"] == dgp_name) & (df["learner_name"] == learner_name)].copy()
    pivot = subset.pivot(index="n", columns="p", values=metric)
    pivot = pivot.reindex(index=N_ORDER, columns=P_ORDER)
    return pivot.to_numpy(dtype=float)


def _plot_heatmaps(df: pd.DataFrame, diagnostics_dir: Path, metric: str) -> list[Path]:
    paths: list[Path] = []
    if metric == "bias":
        vmax = float(np.nanmax(np.abs(df["bias"].to_numpy(dtype=float))))
        vmin = -vmax
        cmap = "RdBu_r"
        title_metric = "Bias"
        suffix = "bias"
        fmt = "{:+.3f}"
    else:
        vmin = 0.0
        vmax = float(np.nanmax(df["abs_bias"].to_numpy(dtype=float)))
        cmap = "YlOrRd"
        title_metric = "Absolute Bias"
        suffix = "abs_bias"
        fmt = "{:.3f}"

    for learner_name in LEARNER_ORDER:
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), sharey=True)
        for idx, dgp_name in enumerate(DGP_ORDER):
            ax = axes[idx]
            matrix = _heatmap_matrix(df, dgp_name=dgp_name, learner_name=learner_name, metric=metric)
            im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            for i, n_value in enumerate(N_ORDER):
                for j, p_value in enumerate(P_ORDER):
                    value = matrix[i, j]
                    ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=8, color="black")
            ax.set_title(DGP_LABELS[dgp_name])
            ax.set_xlabel("p")
            ax.set_xticks(np.arange(len(P_ORDER)))
            ax.set_xticklabels([str(v) for v in P_ORDER])
            ax.set_yticks(np.arange(len(N_ORDER)))
            ax.set_yticklabels([str(v) for v in N_ORDER])
            if idx == 0:
                ax.set_ylabel("n")

        fig.suptitle(f"{title_metric} Heatmaps by DGP: {LEARNER_LABELS[learner_name]}", y=1.02)
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02)
        cbar.set_label(title_metric)
        fig.subplots_adjust(top=0.80, wspace=0.18)

        png_path = diagnostics_dir / f"heatmap_{suffix}_{learner_name}.png"
        pdf_path = diagnostics_dir / f"heatmap_{suffix}_{learner_name}.pdf"
        fig.savefig(png_path)
        fig.savefig(pdf_path)
        plt.close(fig)
        paths.extend([png_path, pdf_path])
    return paths


def _plot_bias_lines(df: pd.DataFrame, diagnostics_dir: Path) -> list[Path]:
    paths: list[Path] = []
    scenario_labels = _scenario_order_labels()
    x = np.arange(len(scenario_labels))

    for dgp_name in DGP_ORDER:
        fig, ax = plt.subplots(figsize=(9.2, 4.2))
        subset = df.loc[df["dgp_name"] == dgp_name].copy()
        for learner_name in LEARNER_ORDER:
            learner_df = subset.loc[subset["learner_name"] == learner_name].copy()
            learner_df["scenario_label"] = pd.Categorical(
                learner_df["scenario_label"], categories=scenario_labels, ordered=True
            )
            learner_df = learner_df.sort_values("scenario_label")
            ax.plot(
                x,
                learner_df["bias"].to_numpy(dtype=float),
                marker=LEARNER_MARKERS[learner_name],
                color=LEARNER_COLORS[learner_name],
                linestyle=LEARNER_LINESTYLES[learner_name],
                linewidth=1.8,
                markersize=5.5,
                label=LEARNER_LABELS[learner_name],
            )

        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_title(f"Bias Across Scenarios: {DGP_LABELS[dgp_name]}")
        ax.set_xlabel("Scenario")
        ax.set_ylabel("Bias")
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_labels, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
        fig.tight_layout()

        stem = dgp_name
        png_path = diagnostics_dir / f"bias_line_{stem}.png"
        pdf_path = diagnostics_dir / f"bias_line_{stem}.pdf"
        fig.savefig(png_path)
        fig.savefig(pdf_path)
        plt.close(fig)
        paths.extend([png_path, pdf_path])
    return paths


def _plot_abs_bias_grouped_bars(df: pd.DataFrame, diagnostics_dir: Path) -> list[Path]:
    paths: list[Path] = []
    scenario_labels = _scenario_order_labels()
    x = np.arange(len(scenario_labels))
    width = 0.24

    for dgp_name in DGP_ORDER:
        fig, ax = plt.subplots(figsize=(9.2, 4.2))
        subset = df.loc[df["dgp_name"] == dgp_name].copy()
        for idx, learner_name in enumerate(LEARNER_ORDER):
            learner_df = subset.loc[subset["learner_name"] == learner_name].copy()
            learner_df["scenario_label"] = pd.Categorical(
                learner_df["scenario_label"], categories=scenario_labels, ordered=True
            )
            learner_df = learner_df.sort_values("scenario_label")
            ax.bar(
                x + (idx - 1) * width,
                learner_df["abs_bias"].to_numpy(dtype=float),
                width=width,
                color=LEARNER_COLORS[learner_name],
                alpha=0.92,
                label=LEARNER_LABELS[learner_name],
            )

        ax.set_title(f"Absolute Bias Across Scenarios: {DGP_LABELS[dgp_name]}")
        ax.set_xlabel("Scenario")
        ax.set_ylabel("Absolute Bias")
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_labels, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
        fig.tight_layout()

        stem = dgp_name
        png_path = diagnostics_dir / f"abs_bias_grouped_bar_{stem}.png"
        pdf_path = diagnostics_dir / f"abs_bias_grouped_bar_{stem}.pdf"
        fig.savefig(png_path)
        fig.savefig(pdf_path)
        plt.close(fig)
        paths.extend([png_path, pdf_path])
    return paths


def _publication_main_table(df: pd.DataFrame, publication_dir: Path) -> Path:
    table_df = df.loc[:, ["dgp_name", "n", "p", "scenario_label", "learner_name", "bias"]].copy()
    table_df["dgp"] = table_df["dgp_name"].map(DGP_LABELS)
    table_df["learner"] = table_df["learner_name"].map(LEARNER_LABELS)
    wide = (
        table_df.pivot_table(
            index=["dgp", "scenario_label", "n", "p"],
            columns="learner",
            values="bias",
            aggfunc="first",
        )
        .reindex(columns=[LEARNER_LABELS[k] for k in LEARNER_ORDER])
        .reset_index()
    )
    wide = wide.rename(columns={"scenario_label": "Scenario", "n": "n", "p": "p"})
    wide = wide.sort_values(["dgp", "n", "p"], kind="mergesort")

    out_path = publication_dir / "table_bias_main_compact.tex"
    latex = _df_to_latex(
        wide,
        float_decimals=4,
        caption="Bias by DGP and simulation scenario.",
        label="tab:bias-1",
    )
    out_path.write_text(latex, encoding="utf-8")
    return out_path


def _publication_appendix_table(df: pd.DataFrame, publication_dir: Path) -> Path:
    appendix = df.loc[:, ["dgp_name", "scenario_label", "n", "p", "learner_name", "bias"]].copy()
    appendix["dgp_name"] = appendix["dgp_name"].map(DGP_LABELS)
    appendix["learner_name"] = appendix["learner_name"].map(LEARNER_LABELS)
    appendix = appendix.rename(
        columns={
            "dgp_name": "DGP",
            "scenario_label": "Scenario",
            "learner_name": "Learner",
            "bias": "Bias",
        }
    )
    appendix = appendix.sort_values(["DGP", "n", "p", "Learner"], kind="mergesort")

    out_path = publication_dir / "table_bias_appendix_full.tex"
    latex = _df_to_latex(
        appendix,
        float_decimals=6,
        caption="Full bias values for all DGP-scenario-learner cells.",
        label="tab:bias-a1",
        longtable=True,
    )
    out_path.write_text(latex, encoding="utf-8")
    return out_path


def _publication_main_figure(df: pd.DataFrame, publication_dir: Path) -> list[Path]:
    scenario_labels = _scenario_order_labels()
    x = np.arange(len(scenario_labels))

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), sharey=True)
    for idx, dgp_name in enumerate(DGP_ORDER):
        ax = axes[idx]
        subset = df.loc[df["dgp_name"] == dgp_name].copy()
        for learner_name in LEARNER_ORDER:
            learner_df = subset.loc[subset["learner_name"] == learner_name].copy()
            learner_df["scenario_label"] = pd.Categorical(
                learner_df["scenario_label"], categories=scenario_labels, ordered=True
            )
            learner_df = learner_df.sort_values("scenario_label")
            ax.plot(
                x,
                learner_df["bias"].to_numpy(dtype=float),
                color=LEARNER_COLORS[learner_name],
                marker=LEARNER_MARKERS[learner_name],
                linestyle=LEARNER_LINESTYLES[learner_name],
                linewidth=2.0,
                markersize=5.6,
                label=LEARNER_LABELS[learner_name],
            )

        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_title(DGP_LABELS[dgp_name])
        ax.set_xlabel("Scenario")
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_labels, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)
        if idx == 0:
            ax.set_ylabel("Bias")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.suptitle("Bias Across Simulation Scenarios by Nuisance Learner", y=1.12)
    fig.tight_layout()

    png_path = publication_dir / "fig_bias_main_two_panel.png"
    pdf_path = publication_dir / "fig_bias_main_two_panel.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path]


def _sign_change_flag(series: pd.Series) -> bool:
    vals = series.to_numpy(dtype=float)
    non_zero = vals[np.abs(vals) > 1e-12]
    if len(non_zero) == 0:
        return False
    return float(non_zero.min()) < 0.0 and float(non_zero.max()) > 0.0


def _write_summary_note(df: pd.DataFrame, notes_dir: Path) -> Path:
    learner_stats = (
        df.groupby("learner_name", as_index=False)
        .agg(
            mean_bias=("bias", "mean"),
            mean_abs_bias=("abs_bias", "mean"),
            max_abs_bias=("abs_bias", "max"),
        )
        .copy()
    )
    learner_stats["learner_name"] = pd.Categorical(
        learner_stats["learner_name"], categories=LEARNER_ORDER, ordered=True
    )
    learner_stats = learner_stats.sort_values("learner_name")

    sign_change = (
        df.groupby("learner_name", as_index=False)["bias"]
        .apply(_sign_change_flag, include_groups=False)
        .rename(columns={"bias": "sign_change"})
    )
    sign_change["learner_name"] = pd.Categorical(
        sign_change["learner_name"], categories=LEARNER_ORDER, ordered=True
    )
    sign_change = sign_change.sort_values("learner_name")

    top_abs_bias = df.sort_values("abs_bias", ascending=False).head(6).copy()
    top_abs_bias["dgp_name"] = top_abs_bias["dgp_name"].map(DGP_LABELS)
    top_abs_bias["learner_name"] = top_abs_bias["learner_name"].map(LEARNER_LABELS)

    lines: list[str] = ["# Bias Section Summary Note", ""]
    lines.append("## Mean bias by learner")
    for row in learner_stats.itertuples(index=False):
        lines.append(f"- {LEARNER_LABELS[row.learner_name]}: {row.mean_bias:+.6f}")
    lines.append("")

    lines.append("## Mean absolute bias by learner")
    for row in learner_stats.itertuples(index=False):
        lines.append(f"- {LEARNER_LABELS[row.learner_name]}: {row.mean_abs_bias:.6f}")
    lines.append("")

    lines.append("## Maximum absolute bias by learner")
    for row in learner_stats.itertuples(index=False):
        lines.append(f"- {LEARNER_LABELS[row.learner_name]}: {row.max_abs_bias:.6f}")
    lines.append("")

    lines.append("## Scenarios with the largest absolute bias")
    for row in top_abs_bias.itertuples(index=False):
        lines.append(
            "- "
            f"{row.dgp_name}, {row.scenario_label}, {row.learner_name}: "
            f"bias={row.bias:+.6f}, abs_bias={row.abs_bias:.6f}"
        )
    lines.append("")

    lines.append("## Bias sign changes across scenarios")
    for row in sign_change.itertuples(index=False):
        value = "yes" if bool(row.sign_change) else "no"
        lines.append(f"- {LEARNER_LABELS[row.learner_name]}: {value}")
    lines.append("")

    out_path = notes_dir / "bias_summary_note.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _write_manifest(paths: list[Path], output_root: Path) -> Path:
    manifest_path = output_root / "manifest_bias_section.txt"
    rels = [path.relative_to(PROJECT_ROOT).as_posix() for path in sorted(paths)]
    manifest_path.write_text("\n".join(rels) + "\n", encoding="utf-8")
    return manifest_path


def generate_bias_section_outputs() -> list[Path]:
    _configure_matplotlib()
    df = _load_bias_data(INPUT_PATH)

    diagnostics_dir = OUTPUT_ROOT / "diagnostics"
    publication_dir = OUTPUT_ROOT / "publication"
    notes_dir = OUTPUT_ROOT / "notes"

    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    publication_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []

    _, _ = _write_pivot_tables(df=df, diagnostics_dir=diagnostics_dir)
    generated.extend(
        [
            diagnostics_dir / "bias_pivot_by_dgp_n_p_x_learner.csv",
            diagnostics_dir / "abs_bias_pivot_by_dgp_n_p_x_learner.csv",
            diagnostics_dir / "bias_pivot_by_dgp_n_p_x_learner.tex",
            diagnostics_dir / "abs_bias_pivot_by_dgp_n_p_x_learner.tex",
        ]
    )

    generated.extend(_plot_heatmaps(df=df, diagnostics_dir=diagnostics_dir, metric="bias"))
    generated.extend(_plot_heatmaps(df=df, diagnostics_dir=diagnostics_dir, metric="abs_bias"))
    generated.extend(_plot_bias_lines(df=df, diagnostics_dir=diagnostics_dir))
    generated.extend(_plot_abs_bias_grouped_bars(df=df, diagnostics_dir=diagnostics_dir))

    generated.append(_publication_main_table(df=df, publication_dir=publication_dir))
    generated.append(_publication_appendix_table(df=df, publication_dir=publication_dir))
    generated.extend(_publication_main_figure(df=df, publication_dir=publication_dir))
    generated.append(_write_summary_note(df=df, notes_dir=notes_dir))

    generated.append(_write_manifest(paths=generated, output_root=OUTPUT_ROOT))
    return generated


if __name__ == "__main__":
    files = generate_bias_section_outputs()
    print(f"Generated {len(files)} bias-section outputs in {OUTPUT_ROOT}")
