"""Lightweight tests for revised workflow/task configuration."""

from __future__ import annotations

import importlib
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


OLD_DGP_NAMES = {
    "linear_baseline",
    "linear_sparse_correlated",
    "linear_dense_independent",
    "nonlinear_smooth",
    "threshold_interaction",
}

TASK_FILES = [
    path
    for path in (Path(__file__).resolve().parents[1] / "tasks").glob("*.py")
    if path.name != "__init__.py"
]


def test_workflow_task_files_do_not_reference_old_dgps() -> None:
    """Task files should not carry old DGP design names."""

    for path in TASK_FILES:
        text = path.read_text(encoding="utf-8")
        for old_name in OLD_DGP_NAMES:
            assert old_name not in text, f"{path} still references {old_name}"


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


@pytest.mark.parametrize("mode", ["pilot", "dev", "fast"])
def test_removed_modes_are_rejected(mode: str) -> None:
    """Removed workflow modes should fail clearly."""

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
    assert config.SMOKE_N_REPLICATIONS == 10
    assert config.N_REPLICATIONS == 1000
    assert get_replication_count("smoke") == config.SMOKE_N_REPLICATIONS
    assert get_replication_count("full") == config.N_REPLICATIONS
    assert output_suffix("smoke") == "_smoke"
    assert output_suffix("full") == ""


def test_task_files_do_not_use_pilot_grid() -> None:
    """Workflow tasks should not use the removed pilot grid."""

    for path in TASK_FILES:
        text = path.read_text(encoding="utf-8")
        assert "PILOT_N_P_PAIRS" not in text, f"{path} still uses pilot grid"


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
