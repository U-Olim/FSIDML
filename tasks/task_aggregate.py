"""Pytask step for aggregation of simulation replications."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    from dml_project import config
    from dml_project.pipeline_mode import get_replication_count, get_run_mode, output_suffix
    from dml_project.simulation.aggregate import aggregate_results
    from dml_project.simulation.scenario_builders import (
        build_main_scenarios,
        count_main_scenarios,
    )
    from dml_project.utils.checks import (
        SCENARIO_SUMMARY_COLUMNS,
        validate_replication_structure,
        validate_scenario_summary_schema,
        validate_summary_structure,
    )
except ModuleNotFoundError:
    from src.dml_project import config
    from src.dml_project.pipeline_mode import get_replication_count, get_run_mode, output_suffix
    from src.dml_project.simulation.aggregate import aggregate_results
    from src.dml_project.simulation.scenario_builders import (
        build_main_scenarios,
        count_main_scenarios,
    )
    from src.dml_project.utils.checks import (
        SCENARIO_SUMMARY_COLUMNS,
        validate_replication_structure,
        validate_scenario_summary_schema,
        validate_summary_structure,
    )

MODE = get_run_mode()
SUFFIX = output_suffix(MODE)


def task_aggregate(
    path_to_raw: Path = PROJECT_ROOT / f"documents/outputs/raw/simulations{SUFFIX}.csv",
    path_to_aggregated: Annotated[Path, Product] = (
        PROJECT_ROOT / f"documents/outputs/aggregated/scenario_summary{SUFFIX}.csv"
    ),
) -> None:
    """Create scenario summary with identifiers and diagnostics."""

    expected_n_rep = get_replication_count(MODE)
    expected_raw = PROJECT_ROOT / f"documents/outputs/raw/simulations{SUFFIX}.csv"
    expected_summary = PROJECT_ROOT / f"documents/outputs/aggregated/scenario_summary{SUFFIX}.csv"
    if path_to_raw != expected_raw:
        raise ValueError(f"task_aggregate must read {expected_raw}, got {path_to_raw}")
    if path_to_aggregated != expected_summary:
        raise ValueError(
            f"task_aggregate must write {expected_summary}, got {path_to_aggregated}"
        )

    raw_df = pd.read_csv(path_to_raw)
    main_scenarios = build_main_scenarios(n_rep=expected_n_rep)
    expected_scenarios = count_main_scenarios()
    if len(main_scenarios) != expected_scenarios:
        raise ValueError(f"Scenario count must be {expected_scenarios}, got {len(main_scenarios)}")

    validate_replication_structure(raw_df, expected_n_rep=expected_n_rep)
    if set(raw_df["scenario_id"]) != {scenario.scenario_id for scenario in main_scenarios}:
        raise ValueError("Raw simulation scenario_id set does not match scenario definitions")

    summary_df = pd.DataFrame(
        [
            {
                "scenario_id": scenario.scenario_id,
                "scenario_name": scenario.name,
                "dgp_name": scenario.dgp_name,
                "learner_name": scenario.learner_name,
                "matched_specification": scenario.matched_specification,
                "n": scenario.n,
                "p": scenario.p,
                "n_rep": scenario.n_rep,
            }
            for scenario in main_scenarios
        ]
    )
    metrics_df = aggregate_results(raw_df)[
        [
            "scenario_id",
            "bias",
            "rmse",
            "empirical_sd",
            "mean_se",
            "coverage",
            "mean_ci_length",
            "variance_ratio",
            "t_stat_mean",
            "t_stat_sd",
        ]
    ]
    summary_df = summary_df.merge(
        metrics_df,
        on="scenario_id",
        how="inner",
        validate="one_to_one",
    ).loc[:, SCENARIO_SUMMARY_COLUMNS]
    required_metrics = {
        "bias",
        "rmse",
        "empirical_sd",
        "mean_se",
        "coverage",
        "mean_ci_length",
        "variance_ratio",
        "t_stat_mean",
        "t_stat_sd",
    }
    missing_metrics = sorted(required_metrics.difference(summary_df.columns))
    if missing_metrics:
        raise ValueError(
            "Aggregated summary is missing required metrics columns: "
            f"{missing_metrics}"
        )
    summary_df["dgp_name"] = pd.Categorical(
        summary_df["dgp_name"], categories=config.DGP_NAMES, ordered=True
    )
    summary_df["learner_name"] = pd.Categorical(
        summary_df["learner_name"], categories=config.LEARNERS, ordered=True
    )
    summary_df = summary_df.sort_values(["dgp_name", "learner_name"]).reset_index(drop=True)
    summary_df["dgp_name"] = summary_df["dgp_name"].astype(str)
    summary_df["learner_name"] = summary_df["learner_name"].astype(str)

    validate_summary_structure(
        summary_df=summary_df,
        expected_scenarios=len(main_scenarios),
        design_key_columns=[
            "scenario_name",
            "dgp_name",
            "learner_name",
            "matched_specification",
            "n",
            "p",
            "n_rep",
        ],
    )
    if summary_df["scenario_id"].duplicated().any():
        raise ValueError("Duplicate scenario_id found after summary aggregation")
    validate_scenario_summary_schema(summary_df)

    path_to_aggregated.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(path_to_aggregated, index=False, encoding="utf-8")
