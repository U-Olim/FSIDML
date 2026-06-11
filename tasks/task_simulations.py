"""Pytask step for full Monte Carlo simulation execution."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Annotated
from typing import Any

import pandas as pd
from joblib import Parallel, delayed
from pytask import Product

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    from dml_project import config
    from dml_project.pipeline_mode import (
        get_replication_count,
        get_run_mode,
        output_suffix,
    )
    from dml_project.simulation.runner import run_scenario_with_runtime
    from dml_project.simulation.scenario_builders import (
        build_scenarios_for_mode,
        count_scenarios_for_mode,
    )
    from dml_project.utils.checks import validate_replication_structure
except ModuleNotFoundError:
    from src.dml_project import config
    from src.dml_project.pipeline_mode import (
        get_replication_count,
        get_run_mode,
        output_suffix,
    )
    from src.dml_project.simulation.runner import run_scenario_with_runtime
    from src.dml_project.simulation.scenario_builders import (
        build_scenarios_for_mode,
        count_scenarios_for_mode,
    )
    from src.dml_project.utils.checks import validate_replication_structure

MODE = get_run_mode()
SUFFIX = output_suffix(MODE)
CPU_COUNT = os.cpu_count() or 1
N_JOBS = min(4, max(1, CPU_COUNT - 1))


class _SimpleScenarioProgress:
    """Minimal scenario-level progress indicator."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.completed = 0
        self.width = 20
        self._render()

    def update(self, step: int = 1) -> None:
        self.completed += step
        self._render()
        if self.completed >= self.total:
            print()

    def _render(self) -> None:
        filled = (
            int(self.width * self.completed / self.total) if self.total else self.width
        )
        bar = "#" * filled + "-" * (self.width - filled)
        print(
            f"\rProgress: [{bar}] {self.completed}/{self.total} scenarios completed",
            end="",
            flush=True,
        )

    def close(self) -> None:
        """Compatibility with tqdm-like progress interfaces."""


def _make_progress(total_scenarios: int) -> Any:
    """Return tqdm progress bar when available, otherwise a simple fallback."""

    try:
        import importlib

        tqdm = importlib.import_module("tqdm.auto").tqdm
    except Exception:  # pragma: no cover - fallback if tqdm is unavailable.
        return _SimpleScenarioProgress(total=total_scenarios)
    return tqdm(total=total_scenarios, desc="Running scenarios", unit="scenario")


def task_simulations(
    path_to_raw: Annotated[Path, Product] = (
        config.RAW_RESULTS_DIR / f"simulations{SUFFIX}.csv"
    ),
) -> None:
    """Run simulations and write replication-level outputs with integrity checks."""
    mode = MODE
    suffix = SUFFIX
    expected_path = config.RAW_RESULTS_DIR / f"simulations{suffix}.csv"
    if path_to_raw != expected_path:
        raise ValueError(
            f"task_simulations must write to {expected_path}, got {path_to_raw}"
        )

    expected_n_rep = get_replication_count(mode)
    scenarios = build_scenarios_for_mode(mode, n_rep=expected_n_rep)
    expected_scenarios = count_scenarios_for_mode(mode)
    if len(scenarios) != expected_scenarios:
        raise ValueError(
            f"Scenario count must be {expected_scenarios}, got {len(scenarios)}"
        )

    print(f"Mode: {mode}, scenarios: {len(scenarios)}, n_rep={expected_n_rep}")

    path_to_raw.parent.mkdir(parents=True, exist_ok=True)
    if path_to_raw.exists():
        path_to_raw.unlink()
    runtime_stats_path = (
        config.AGGREGATED_RESULTS_DIR / f"scenario_runtime_stats{suffix}.csv"
    )
    runtime_stats_path.parent.mkdir(parents=True, exist_ok=True)
    if runtime_stats_path.exists():
        runtime_stats_path.unlink()

    results_list: list[pd.DataFrame] = []
    runtime_header_written = False
    total_scenarios = len(scenarios)
    progress = _make_progress(total_scenarios)
    parallel = Parallel(n_jobs=N_JOBS, return_as="generator")
    try:
        for completed_count, scenario_output in enumerate(
            parallel(
                delayed(run_scenario_with_runtime)(scenario, mode)
                for scenario in scenarios
            ),
            start=1,
        ):
            if scenario_output is None:
                continue
            result, runtime_row = scenario_output
            results_list.append(result)
            runtime_df = pd.DataFrame([runtime_row])
            runtime_df.to_csv(
                runtime_stats_path,
                mode="a",
                header=not runtime_header_written,
                index=False,
                encoding="utf-8",
            )
            runtime_header_written = True
            partial_df = pd.concat(results_list, ignore_index=True)
            partial_df.to_csv(path_to_raw, index=False, encoding="utf-8")
            progress.update(1)
            print(
                f"Scenario {runtime_row['scenario_id']} finished | "
                f"DGP={runtime_row['dgp_name']} | "
                f"learner={runtime_row['learner_name']} | "
                f"n={runtime_row['n']} | p={runtime_row['p']} | "
                f"duration={runtime_row['duration_seconds']:.2f}s"
            )
            print(f"Completed scenario {completed_count} / {len(scenarios)}")
    finally:
        progress.close()

    results = pd.concat(results_list, ignore_index=True)
    validate_replication_structure(results, expected_n_rep=expected_n_rep)

    expected_by_scenario = {
        scenario.scenario_id: scenario.n_rep for scenario in scenarios
    }
    counts = results.groupby("scenario_id").size().to_dict()
    if counts != expected_by_scenario:
        raise ValueError(
            "Simulation output counts do not match scenario definitions. "
            f"expected={expected_by_scenario}, got={counts}"
        )

    results.to_csv(path_to_raw, index=False, encoding="utf-8")
