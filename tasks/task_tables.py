"""Pytask step for publication-quality table generation."""

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
    from dml_project.utils.checks import (
        MAIN_RESULTS_COLUMNS,
        validate_main_results_schema,
        validate_summary_and_results_alignment,
    )
except ModuleNotFoundError:
    from src.dml_project import config
    from src.dml_project.pipeline_mode import get_run_mode, output_suffix
    from src.dml_project.utils.checks import (
        MAIN_RESULTS_COLUMNS,
        validate_main_results_schema,
        validate_summary_and_results_alignment,
    )

MODE = get_run_mode()
SUFFIX = output_suffix(MODE)

_DGP_FILE_STEMS = {
    "linear_baseline": "linear_baseline",
    "linear_sparse_correlated": "linear_sparse_correlated",
}
_DGP_PANEL_NAMES = {
    "linear_baseline": "Linear Baseline DGP",
    "linear_sparse_correlated": "Linear Sparse Correlated DGP",
}
_LEARNER_COLUMNS = list(config.LEARNERS)


def _ordered_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    ordered = summary_df.copy()
    ordered["dgp_name"] = pd.Categorical(
        ordered["dgp_name"], categories=config.DGP_NAMES, ordered=True
    )
    ordered["learner_name"] = pd.Categorical(
        ordered["learner_name"], categories=config.LEARNERS, ordered=True
    )
    ordered["p"] = pd.Categorical(ordered["p"], categories=config.P_VALUES, ordered=True)
    ordered["n"] = pd.Categorical(ordered["n"], categories=config.N_VALUES, ordered=True)
    ordered = ordered.sort_values(["dgp_name", "p", "n", "learner_name"]).reset_index(drop=True)
    ordered["dgp_name"] = ordered["dgp_name"].astype(str)
    ordered["learner_name"] = ordered["learner_name"].astype(str)
    ordered["p"] = ordered["p"].astype(int)
    ordered["n"] = ordered["n"].astype(int)
    return ordered


def _build_main_results(summary_df: pd.DataFrame) -> pd.DataFrame:
    table_df = summary_df.loc[:, MAIN_RESULTS_COLUMNS].copy()
    table_df["dgp_name"] = pd.Categorical(
        table_df["dgp_name"], categories=config.DGP_NAMES, ordered=True
    )
    table_df["learner_name"] = pd.Categorical(
        table_df["learner_name"], categories=config.LEARNERS, ordered=True
    )
    table_df = table_df.sort_values(["dgp_name", "learner_name", "n", "p"]).reset_index(drop=True)
    table_df["dgp_name"] = table_df["dgp_name"].astype(str)
    table_df["learner_name"] = table_df["learner_name"].astype(str)
    return table_df


def _metric_wide_by_dgp(summary_df: pd.DataFrame, dgp_name: str, metric: str) -> pd.DataFrame:
    subset = summary_df.loc[summary_df["dgp_name"] == dgp_name, ["p", "n", "learner_name", metric]].copy()
    wide = subset.pivot(index=["p", "n"], columns="learner_name", values=metric)
    wide = wide.reindex(
        index=pd.MultiIndex.from_product([config.P_VALUES, config.N_VALUES], names=["p", "n"]),
        columns=_LEARNER_COLUMNS,
    )
    wide = wide.reset_index()
    return wide.round(3)


def _rmse_bias_table_by_dgp(summary_df: pd.DataFrame, dgp_name: str) -> pd.DataFrame:
    subset = summary_df.loc[
        summary_df["dgp_name"] == dgp_name, ["p", "n", "learner_name", "bias", "rmse"]
    ].copy()
    subset["abs_bias"] = subset["bias"].abs()
    wide = subset.pivot(index=["p", "n"], columns="learner_name", values=["bias", "abs_bias", "rmse"])
    wide = wide.reindex(
        index=pd.MultiIndex.from_product([config.P_VALUES, config.N_VALUES], names=["p", "n"]),
        columns=pd.MultiIndex.from_product([["bias", "abs_bias", "rmse"], _LEARNER_COLUMNS]),
    )
    wide.columns = [f"{metric}_{learner}" for metric, learner in wide.columns]
    return wide.reset_index().round(3)


def _build_simulation_design_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    design = summary_df[
        ["scenario_id", "dgp_name", "p", "n", "learner_name", "n_rep", "scenario_name"]
    ].copy()
    design["theta"] = config.THETA_TRUE
    design["seed_base"] = config.BASE_SEED
    design = design.rename(
        columns={
            "dgp_name": "dgp",
            "learner_name": "learner",
            "n_rep": "R",
            "scenario_name": "scenario",
        }
    )
    return design[
        ["scenario_id", "scenario", "dgp", "p", "n", "learner", "R", "theta", "seed_base"]
    ].reset_index(drop=True)


def _build_learner_ranking_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    base = summary_df[["dgp_name", "p", "n", "learner_name", "coverage", "rmse", "bias"]].copy()
    base["abs_bias"] = base["bias"].abs()

    coverage_winners = base.groupby(["dgp_name", "p", "n"])["coverage"].transform("max") == base["coverage"]
    rmse_winners = base.groupby(["dgp_name", "p", "n"])["rmse"].transform("min") == base["rmse"]
    abs_bias_winners = (
        base.groupby(["dgp_name", "p", "n"])["abs_bias"].transform("min") == base["abs_bias"]
    )

    counts = (
        pd.DataFrame(
            {
                "learner": base["learner_name"],
                "best_coverage_count": coverage_winners.astype(int),
                "lowest_rmse_count": rmse_winners.astype(int),
                "lowest_abs_bias_count": abs_bias_winners.astype(int),
            }
        )
        .groupby("learner", as_index=False)
        .sum()
    )
    counts["learner"] = pd.Categorical(counts["learner"], categories=config.LEARNERS, ordered=True)
    counts = counts.sort_values("learner").reset_index(drop=True)
    counts["learner"] = counts["learner"].astype(str)
    counts["total_scenarios"] = len(config.DGP_NAMES) * len(config.P_VALUES) * len(config.N_VALUES)
    return counts


def _to_markdown_table(df: pd.DataFrame) -> str:
    headers = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _coverage_markdown_with_bold_and_nominal(wide_df: pd.DataFrame) -> str:
    nominal = 0.95
    formatted_rows: list[dict[str, str | int]] = []
    for _, row in wide_df.iterrows():
        learners = {learner: float(row[learner]) for learner in _LEARNER_COLUMNS}
        row_best = max(learners.values())
        rendered: dict[str, str | int] = {"p": int(row["p"]), "n": int(row["n"])}
        for learner in _LEARNER_COLUMNS:
            value = learners[learner]
            text = f"{value:.3f}"
            if abs(value - row_best) < 1e-12:
                text = f"**{text}**"
            if abs(value - nominal) <= 0.01:
                text = f"{text}*"
            rendered[learner] = text
        formatted_rows.append(rendered)
    markdown_df = pd.DataFrame(formatted_rows)
    return _to_markdown_table(markdown_df) + (
        "\n\nNominal coverage is 0.95. Bold marks best coverage per row. "
        "* marks values within +/-0.01 of 0.95.\n"
    )


def _write_coverage_html_heatmap(wide_df: pd.DataFrame, output_path: Path) -> None:
    coverage_only = wide_df.set_index(["p", "n"])[_LEARNER_COLUMNS].copy()
    row_max = coverage_only.eq(coverage_only.max(axis=1), axis=0)
    close_nominal = (coverage_only - 0.95).abs() <= 0.01
    html_lines = [
        "<html><head><meta charset='utf-8'><title>Coverage Heatmap</title></head><body>",
        "<h3>Empirical coverage (nominal 0.95)</h3>",
        "<p>Bold = best per row. Thick border = within +/-0.01 of 0.95.</p>",
        "<table border='1' cellspacing='0' cellpadding='6'>",
        "<tr><th>p</th><th>n</th><th>ols</th><th>lasso</th><th>elastic_net</th></tr>",
    ]

    def _cell_color(value: float) -> str:
        clipped = min(max(value, 0.0), 1.0)
        if clipped <= 0.5:
            ratio = clipped / 0.5
            red, green, blue = 255, int(255 * ratio), 140
        else:
            ratio = (clipped - 0.5) / 0.5
            red, green, blue = int(255 * (1.0 - ratio)), 255, 140
        return f"rgb({red},{green},{blue})"

    def _as_float_scalar(value) -> float:
        """Convert pandas cell values to a single float for HTML rendering."""

        scalar = np.asarray(value).reshape(-1)[0]
        return float(pd.to_numeric(scalar, errors="coerce"))

    for p_value, n_value in coverage_only.index:
        html_lines.append(f"<tr><td>{int(p_value)}</td><td>{int(n_value)}</td>")
        for learner in _LEARNER_COLUMNS:
            value = _as_float_scalar(coverage_only.at[(p_value, n_value), learner])
            styles = [f"background-color:{_cell_color(value)}"]
            if bool(row_max.loc[(p_value, n_value), learner]):
                styles.append("font-weight:700")
            if bool(close_nominal.loc[(p_value, n_value), learner]):
                styles.append("border:2px solid #1f2937")
            style_attr = ";".join(styles)
            html_lines.append(f"<td style='{style_attr}'>{value:.3f}</td>")
        html_lines.append("</tr>")

    html_lines.extend(["</table>", "</body></html>"])
    output_path.write_text("\n".join(html_lines), encoding="utf-8")


def _write_panel_markdown(
    *,
    output_path: Path,
    panel_title: str,
    body_markdown: str,
    note: str | None = None,
) -> None:
    lines = [f"# {panel_title}", "", body_markdown]
    if note:
        lines.extend(["", note])
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def task_tables(
    path_to_raw: Path = PROJECT_ROOT / f"documents/outputs/raw/simulations{SUFFIX}.csv",
    path_to_summary: Path = PROJECT_ROOT / f"documents/outputs/aggregated/scenario_summary{SUFFIX}.csv",
    path_to_table: Annotated[Path, Product] = (
        PROJECT_ROOT / f"documents/outputs/tables/main_results{SUFFIX}.csv"
    ),
    path_to_table_manifest: Annotated[Path, Product] = (
        PROJECT_ROOT / f"documents/outputs/tables/table_suite_manifest{SUFFIX}.txt"
    ),
) -> None:
    """Create main results plus publication-ready scenario and metric table suite."""

    expected_raw = PROJECT_ROOT / f"documents/outputs/raw/simulations{SUFFIX}.csv"
    expected_summary = PROJECT_ROOT / f"documents/outputs/aggregated/scenario_summary{SUFFIX}.csv"
    expected_table = PROJECT_ROOT / f"documents/outputs/tables/main_results{SUFFIX}.csv"
    if path_to_raw != expected_raw:
        raise ValueError(f"task_tables must read {expected_raw}, got {path_to_raw}")
    if path_to_summary != expected_summary:
        raise ValueError(f"task_tables must read {expected_summary}, got {path_to_summary}")
    if path_to_table != expected_table:
        raise ValueError(f"task_tables must write {expected_table}, got {path_to_table}")

    _ = pd.read_csv(path_to_raw)
    summary_df = _ordered_summary(pd.read_csv(path_to_summary))

    table_df = _build_main_results(summary_df)
    validate_main_results_schema(table_df)
    validate_summary_and_results_alignment(summary_df=summary_df, main_results_df=table_df)
    path_to_table.parent.mkdir(parents=True, exist_ok=True)
    table_df.to_csv(path_to_table, index=False, encoding="utf-8")

    table_dir = path_to_table.parent
    generated: list[Path] = [path_to_table]

    design_table = _build_simulation_design_table(summary_df)
    design_csv = table_dir / f"simulation_design_table{SUFFIX}.csv"
    design_md = table_dir / f"simulation_design_table{SUFFIX}.md"
    design_table.to_csv(design_csv, index=False, encoding="utf-8")
    _write_panel_markdown(
        output_path=design_md,
        panel_title="Simulation Design and Scenario Summary",
        body_markdown=_to_markdown_table(design_table),
    )
    generated.extend([design_csv, design_md])

    for dgp_name in config.DGP_NAMES:
        dgp_stem = _DGP_FILE_STEMS[dgp_name]
        panel_name = _DGP_PANEL_NAMES[dgp_name]

        coverage_table = _metric_wide_by_dgp(summary_df, dgp_name=dgp_name, metric="coverage")
        coverage_csv = table_dir / f"coverage_table_{dgp_stem}{SUFFIX}.csv"
        coverage_md = table_dir / f"coverage_table_{dgp_stem}{SUFFIX}.md"
        coverage_html = table_dir / f"coverage_table_{dgp_stem}_heatmap{SUFFIX}.html"
        coverage_table.to_csv(coverage_csv, index=False, encoding="utf-8")
        _write_panel_markdown(
            output_path=coverage_md,
            panel_title=f"Coverage Table ({panel_name})",
            body_markdown=_coverage_markdown_with_bold_and_nominal(coverage_table),
            note="Nominal coverage target is 0.95.",
        )
        _write_coverage_html_heatmap(coverage_table, coverage_html)
        generated.extend([coverage_csv, coverage_md, coverage_html])

        rmse_bias_table = _rmse_bias_table_by_dgp(summary_df, dgp_name=dgp_name)
        rmse_bias_csv = table_dir / f"rmse_bias_table_{dgp_stem}{SUFFIX}.csv"
        rmse_bias_md = table_dir / f"rmse_bias_table_{dgp_stem}{SUFFIX}.md"
        rmse_bias_table.to_csv(rmse_bias_csv, index=False, encoding="utf-8")
        _write_panel_markdown(
            output_path=rmse_bias_md,
            panel_title=f"Bias and RMSE Table ({panel_name})",
            body_markdown=_to_markdown_table(rmse_bias_table),
            note="Bias is signed. Absolute bias is included as abs_bias_* columns.",
        )
        generated.extend([rmse_bias_csv, rmse_bias_md])

        ci_length_table = _metric_wide_by_dgp(summary_df, dgp_name=dgp_name, metric="mean_ci_length")
        ci_csv = table_dir / f"ci_length_table_{dgp_stem}{SUFFIX}.csv"
        ci_md = table_dir / f"ci_length_table_{dgp_stem}{SUFFIX}.md"
        ci_length_table.to_csv(ci_csv, index=False, encoding="utf-8")
        _write_panel_markdown(
            output_path=ci_md,
            panel_title=f"CI Length Table ({panel_name})",
            body_markdown=_to_markdown_table(ci_length_table),
        )
        generated.extend([ci_csv, ci_md])

    ranking = _build_learner_ranking_summary(summary_df)
    ranking_csv = table_dir / f"learner_ranking_summary{SUFFIX}.csv"
    ranking_md = table_dir / f"learner_ranking_summary{SUFFIX}.md"
    ranking.to_csv(ranking_csv, index=False, encoding="utf-8")
    _write_panel_markdown(
        output_path=ranking_md,
        panel_title="Learner Ranking Summary",
        body_markdown=_to_markdown_table(ranking),
        note=(
            "Counts are based on scenario cells (DGP, p, n). Ties count for all tied learners."
        ),
    )
    generated.extend([ranking_csv, ranking_md])

    manifest_lines = [
        str(path.relative_to(PROJECT_ROOT).as_posix())
        for path in sorted(set(generated), key=lambda path: str(path))
    ]
    path_to_table_manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
