"""Lightweight tests for revised workflow/task configuration."""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest

from dml_project import config
from dml_project.pipeline_mode import (
    VALID_RUN_MODES,
    get_replication_count,
    output_suffix,
)
from dml_project.simulation.scenario_builders import (
    build_all_scenarios,
    build_scenarios_for_mode,
    build_smoke_scenarios,
)


TASK_FILES = [
    path
    for path in (Path(__file__).resolve().parents[1] / "tasks").glob("*.py")
    if path.name != "__init__.py"
]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_TASK_MODULES = {
    "task_aggregate.py",
    "task_figures.py",
    "task_simulations.py",
    "task_tables.py",
}


def test_only_simulation_paper_tasks_are_present() -> None:
    """Workflow should expose only active simulation-paper task files."""

    assert {path.name for path in TASK_FILES} == ACTIVE_TASK_MODULES


def test_workflow_task_files_use_current_dgp_config() -> None:
    """Task files should depend on the configured DGP set."""

    figure_text = (PROJECT_ROOT / "tasks" / "task_figures.py").read_text(encoding="utf-8")
    table_text = (PROJECT_ROOT / "tasks" / "task_tables.py").read_text(encoding="utf-8")

    assert "config.DGP_NAMES" in figure_text
    assert "config.DGP_NAMES" in table_text


def test_workflow_task_files_do_not_use_n_folds_constant() -> None:
    """Workflow tasks should use scenario-specific folds, not config.N_FOLDS."""

    for path in TASK_FILES:
        text = path.read_text(encoding="utf-8")
        assert "N_FOLDS" not in text, f"{path} still references N_FOLDS"


def _expected_full_scenario_count() -> int:
    return (
        len(config.N_VALUES)
        * len(config.P_VALUES)
        * len(config.K_VALUES)
        * len(config.DGP_NAMES)
        * len(config.LEARNERS)
    )


def test_allowed_modes_are_smoke_and_full() -> None:
    """Workflow should expose only the two supported simulation modes."""

    assert VALID_RUN_MODES == {"smoke", "full"}


def test_project_config_exposes_no_obsolete_workflow_aliases() -> None:
    """Makefile and pixi tasks should not expose removed workflow aliases."""

    makefile_text = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    pixi_config = tomllib.loads((PROJECT_ROOT / "pixi.toml").read_text())
    pixi_tasks = pixi_config["tasks"]
    config_text = makefile_text + "\n" + (PROJECT_ROOT / "pixi.toml").read_text(encoding="utf-8")

    for obsolete_mode in ("fast", "pilot", "dev"):
        assert obsolete_mode not in config_text
    assert "pytask" in pixi_tasks


@pytest.mark.parametrize("mode", ["invalid", "debug", "quick"])
def test_unknown_modes_are_rejected(mode: str) -> None:
    """Unsupported workflow modes should fail clearly."""

    with pytest.raises(ValueError, match="Allowed modes are smoke and full"):
        build_scenarios_for_mode(mode)
    with pytest.raises(ValueError, match="Allowed modes are smoke and full"):
        get_replication_count(mode)
    with pytest.raises(ValueError, match="Allowed modes are smoke and full"):
        output_suffix(mode)


def test_smoke_scenario_builder_returns_full_grid() -> None:
    """Smoke workflow should use the complete scenario grid."""

    scenarios = build_smoke_scenarios()

    assert len(scenarios) == _expected_full_scenario_count()
    assert all(scenario.n_rep == config.SMOKE_N_REPLICATIONS for scenario in scenarios)


def test_full_scenario_builder_count_matches_design() -> None:
    """Full workflow should use the complete n/p/K/DGP/learner grid."""

    scenarios = build_all_scenarios()
    expected = _expected_full_scenario_count()

    assert len(scenarios) == expected
    assert all(scenario.n_rep == config.N_REPLICATIONS for scenario in scenarios)


def test_smoke_and_full_have_same_scenario_names() -> None:
    """Smoke and full modes should differ only by replication count."""

    smoke_names = [scenario.name for scenario in build_scenarios_for_mode("smoke")]
    full_names = [scenario.name for scenario in build_scenarios_for_mode("full")]

    assert smoke_names == full_names


def test_mode_specific_builders_and_replication_counts() -> None:
    """Workflow modes should select smoke and full grids explicitly."""

    assert len(build_scenarios_for_mode("smoke")) == _expected_full_scenario_count()
    assert len(build_scenarios_for_mode("full")) == len(build_all_scenarios())
    assert config.SMOKE_N_REPLICATIONS == 100
    assert config.N_REPLICATIONS == 1000
    assert get_replication_count("smoke") == config.SMOKE_N_REPLICATIONS
    assert get_replication_count("full") == config.N_REPLICATIONS
    assert output_suffix("smoke") == "_smoke"
    assert output_suffix("full") == ""


def test_task_files_do_not_use_removed_grid_constant() -> None:
    """Workflow tasks should not use a removed grid constant."""

    for path in TASK_FILES:
        text = path.read_text(encoding="utf-8")
        assert "PILOT" not in text, f"{path} still uses a removed grid constant"


def test_pytask_ignores_pytest_temp_directories() -> None:
    """pytask should not collect pytest temp/cache folders on Windows."""

    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    ignore = pyproject["tool"]["pytask"]["ini_options"]["ignore"]

    assert ".pytest_tmp" in ignore
    assert ".pytest_cache" in ignore
    assert "__pycache__" in ignore


def test_workflow_output_paths_construct_without_running() -> None:
    """Central output directories should construct expected workflow paths."""

    paths = [
        config.RAW_RESULTS_DIR / "simulations_smoke.csv",
        config.AGGREGATED_RESULTS_DIR / "scenario_summary_smoke.csv",
        config.TABLES_DIR / "table_main_results_smoke.csv",
        config.FIGURES_DIR / "figure_coverage_by_fold_ratio_smoke.png",
    ]

    assert all(path.is_absolute() for path in paths)
    assert all("documents" in path.parts and "outputs" in path.parts for path in paths)


def test_task_modules_import_without_running_simulations() -> None:
    """Workflow modules should import without executing simulations."""

    module_names = [
        "tasks.task_aggregate",
        "tasks.task_simulations",
        "tasks.task_tables",
    ]
    try:
        import matplotlib  # noqa: F401
    except ModuleNotFoundError:
        pass
    else:
        module_names.append("tasks.task_figures")

    for module_name in module_names:
        importlib.import_module(module_name)
