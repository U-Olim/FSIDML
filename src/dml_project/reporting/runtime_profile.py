"""Runtime profiling tables for simulation outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dml_project import config

RUNTIME_COLUMNS = [
    "runtime_seconds",
    "elapsed_seconds",
    "fit_time",
    "duration",
    "duration_seconds",
    "scenario_runtime",
    "scenario_runtime_seconds",
    "learner_runtime",
    "learner_runtime_seconds",
]

SCENARIO_ID_COLUMNS = [
    "scenario_id",
    "scenario_name",
    "dgp_name",
    "learner_name",
    "n_obs",
    "n_covariates",
    "n",
    "p",
    "n_folds",
]


def _runtime_column(df: pd.DataFrame) -> str | None:
    """Return the first recognized runtime column."""

    for column in RUNTIME_COLUMNS:
        if column in df.columns:
            return column
    return None


def _with_design_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with n_obs/n_covariates aliases populated when possible."""

    prepared = df.copy()
    if "n_obs" not in prepared.columns and "n" in prepared.columns:
        prepared["n_obs"] = prepared["n"]
    if "n_covariates" not in prepared.columns and "p" in prepared.columns:
        prepared["n_covariates"] = prepared["p"]
    return prepared


def _scenario_key(df: pd.DataFrame) -> list[str]:
    """Return the best available scenario key columns."""

    if "scenario_id" in df.columns:
        return ["scenario_id"]
    if "scenario_name" in df.columns:
        return ["scenario_name"]
    fallback = [
        column
        for column in ["dgp_name", "learner_name", "n_obs", "n_covariates", "n_folds"]
        if column in df.columns
    ]
    if fallback:
        return fallback
    raise ValueError("Cannot identify scenarios without scenario_id, scenario_name, or design columns")


def _companion_runtime_path(raw_results_path: Path) -> Path | None:
    """Return the companion scenario-runtime file when it exists."""

    stem = raw_results_path.stem
    suffix = ""
    if stem.startswith("simulations"):
        suffix = stem.removeprefix("simulations")
    candidates = [
        raw_results_path.parents[1]
        / "aggregated"
        / f"scenario_runtime_stats{suffix}.csv",
        raw_results_path.with_name(f"scenario_runtime_stats{suffix}.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _merge_companion_runtime(raw_df: pd.DataFrame, raw_results_path: Path) -> pd.DataFrame | None:
    """Merge scenario-level runtime stats onto raw scenario design columns."""

    runtime_path = _companion_runtime_path(raw_results_path)
    if runtime_path is None:
        return None
    runtime_df = _with_design_aliases(pd.read_csv(runtime_path))
    runtime_column = _runtime_column(runtime_df)
    if runtime_column is None:
        return None

    raw_design = _with_design_aliases(raw_df)
    key = _scenario_key(raw_design)
    design_columns = [
        column
        for column in SCENARIO_ID_COLUMNS
        if column in raw_design.columns and column not in key
    ]
    design_df = raw_design[key + design_columns].drop_duplicates(subset=key)
    merged = runtime_df.merge(
        design_df,
        on=key,
        how="left",
        suffixes=("", "_raw"),
        validate="one_to_one",
    )
    for column in ["dgp_name", "learner_name", "n_obs", "n_covariates", "n_folds"]:
        raw_column = f"{column}_raw"
        if raw_column in merged.columns:
            if column in merged.columns:
                merged[column] = merged[column].fillna(merged[raw_column])
            else:
                merged[column] = merged[raw_column]
            merged = merged.drop(columns=[raw_column])
    return _with_design_aliases(merged)


def _runtime_summary(
    df: pd.DataFrame,
    group_columns: list[str],
    runtime_column: str,
) -> pd.DataFrame:
    """Summarize runtime by the requested grouping columns."""

    if any(column not in df.columns for column in group_columns):
        return pd.DataFrame()
    grouped = df.groupby(group_columns, dropna=False, sort=True)
    scenario_key = _scenario_key(df)
    summary = grouped.agg(
        n_rows=(runtime_column, "size"),
        total_runtime_seconds=(runtime_column, "sum"),
        mean_runtime_seconds=(runtime_column, "mean"),
        median_runtime_seconds=(runtime_column, "median"),
        max_runtime_seconds=(runtime_column, "max"),
    ).reset_index()
    scenario_counts = (
        df[group_columns + scenario_key]
        .drop_duplicates()
        .groupby(group_columns, dropna=False, sort=True)
        .size()
        .rename("n_scenarios")
    )
    return summary.merge(
        scenario_counts.reset_index(),
        on=group_columns,
        how="left",
    )[
        [
            *group_columns,
            "n_rows",
            "n_scenarios",
            "total_runtime_seconds",
            "mean_runtime_seconds",
            "median_runtime_seconds",
            "max_runtime_seconds",
        ]
    ]


def _count_summary(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Summarize row and scenario counts by grouping columns."""

    if any(column not in df.columns for column in group_columns):
        return pd.DataFrame()
    grouped = df.groupby(group_columns, dropna=False, sort=True)
    scenario_key = _scenario_key(df)
    summary = grouped.size().rename("n_rows")
    scenario_counts = (
        df[group_columns + scenario_key]
        .drop_duplicates()
        .groupby(group_columns, dropna=False, sort=True)
        .size()
        .rename("n_scenarios")
    )
    return (
        summary.reset_index()
        .merge(scenario_counts.reset_index(), on=group_columns, how="left")
        .loc[:, [*group_columns, "n_rows", "n_scenarios"]]
    )


def _scenario_completion_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return one-row summary of replication completeness by scenario."""

    scenario_key = _scenario_key(df)
    if "replication" in df.columns:
        replication_counts = df.groupby(scenario_key, dropna=False)["replication"].nunique()
        inferred_expected = int(pd.to_numeric(df["replication"], errors="coerce").max()) + 1
    elif "n_rep" in df.columns:
        replication_counts = df.groupby(scenario_key, dropna=False)["n_rep"].max()
        inferred_expected = int(pd.to_numeric(df["n_rep"], errors="coerce").max())
    else:
        replication_counts = df.groupby(scenario_key, dropna=False).size()
        inferred_expected = int(replication_counts.max())

    expected = config.SMOKE_N_REPLICATIONS
    if expected <= 0:
        expected = inferred_expected
    incomplete = replication_counts < expected
    return pd.DataFrame(
        [
            {
                "total_rows": int(len(df)),
                "unique_scenarios": int(replication_counts.shape[0]),
                "expected_replications_per_scenario": int(expected),
                "inferred_replications_per_scenario": int(inferred_expected),
                "min_replications_per_scenario": int(replication_counts.min()),
                "median_replications_per_scenario": float(replication_counts.median()),
                "max_replications_per_scenario": int(replication_counts.max()),
                "incomplete_scenarios_count": int(incomplete.sum()),
            }
        ]
    )


def _slowest_scenarios(df: pd.DataFrame, runtime_column: str) -> pd.DataFrame:
    """Return the 25 slowest scenario rows or groups."""

    scenario_key = _scenario_key(df)
    label_columns = [
        column
        for column in [
            "scenario_name",
            "dgp_name",
            "learner_name",
            "n_obs",
            "n_covariates",
            "n_folds",
        ]
        if column in df.columns and column not in scenario_key
    ]
    grouped = (
        df.groupby(scenario_key + label_columns, dropna=False, as_index=False)[runtime_column]
        .sum()
        .rename(columns={runtime_column: "total_runtime_seconds"})
        .sort_values("total_runtime_seconds", ascending=False)
        .head(25)
        .reset_index(drop=True)
    )
    return grouped


def create_runtime_profile(raw_results_path: str | Path) -> dict[str, pd.DataFrame]:
    """Create runtime or completion-count profile tables from simulation outputs."""

    path = Path(raw_results_path)
    raw_df = _with_design_aliases(pd.read_csv(path))
    completion = _scenario_completion_summary(raw_df)

    profile_df = raw_df
    runtime_column = _runtime_column(profile_df)
    if runtime_column is None:
        companion_df = _merge_companion_runtime(raw_df=raw_df, raw_results_path=path)
        if companion_df is not None:
            profile_df = companion_df
            runtime_column = _runtime_column(profile_df)

    if runtime_column is None:
        return {
            "rows_by_learner": _count_summary(raw_df, ["learner_name"]),
            "rows_by_dgp": _count_summary(raw_df, ["dgp_name"]),
            "rows_by_k": _count_summary(raw_df, ["n_folds"]),
            "rows_by_n_p": _count_summary(raw_df, ["n_obs", "n_covariates"]),
            "rows_by_learner_and_k": _count_summary(raw_df, ["learner_name", "n_folds"]),
            "scenario_completion_summary": completion,
        }

    return {
        "runtime_by_learner": _runtime_summary(profile_df, ["learner_name"], runtime_column),
        "runtime_by_dgp": _runtime_summary(profile_df, ["dgp_name"], runtime_column),
        "runtime_by_k": _runtime_summary(profile_df, ["n_folds"], runtime_column),
        "runtime_by_n_p": _runtime_summary(
            profile_df,
            ["n_obs", "n_covariates"],
            runtime_column,
        ),
        "runtime_by_learner_and_k": _runtime_summary(
            profile_df,
            ["learner_name", "n_folds"],
            runtime_column,
        ),
        "slowest_scenarios": _slowest_scenarios(profile_df, runtime_column),
        "scenario_completion_summary": completion,
    }
