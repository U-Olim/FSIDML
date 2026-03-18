"""Tests for simulation scenario datatypes and builders."""

import pandas as pd
import pytest

from dml_project import config
from dml_project.pipeline_mode import get_replication_count, get_run_mode
from dml_project.simulation.runner import run_scenario, run_single_replication
from dml_project.simulation.scenario import Scenario, make_scenario_name
from dml_project.simulation.scenario_builders import build_main_scenarios, count_main_scenarios

TEST_N_REP = config.FULL_N_REP


def test_scenario_valid_instance() -> None:
    """A valid Scenario should be created successfully."""
    scenario = Scenario(
        scenario_id=0,
        name="valid",
        dgp_name="linear_baseline",
        learner_name="lasso",
        n=100,
        p=150,
        theta=config.THETA_TRUE,
        n_rep=TEST_N_REP,
        base_seed=config.BASE_SEED,
    )
    assert scenario.scenario_id == 0
    assert scenario.to_dict()["name"] == "valid"
    assert scenario.matched_specification is True
    assert (
        scenario.short_label()
        == f"linear_baseline_lasso_n100_p150_th{config.THETA_TRUE}"
    )


@pytest.mark.parametrize(
    ("dgp_name", "learner_name", "expected"),
    [
        ("linear_baseline", "ols", True),
        ("linear_baseline", "lasso", True),
        ("linear_baseline", "elastic_net", True),
        ("linear_sparse_correlated", "ols", True),
        ("linear_sparse_correlated", "lasso", True),
        ("linear_sparse_correlated", "elastic_net", True),
    ],
)
def test_matched_specification_reflects_linear_model_compatibility(
    dgp_name: str, learner_name: str, expected: bool
) -> None:
    """Specification flag should mark only linear DGP rows as matched."""
    scenario = Scenario(
        scenario_id=99,
        name="compatibility_check",
        dgp_name=dgp_name,
        learner_name=learner_name,
        n=100,
        p=150,
        theta=config.THETA_TRUE,
        n_rep=10,
        base_seed=config.BASE_SEED,
    )
    assert scenario.matched_specification is expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scenario_id": -1},
        {"n": 0},
        {"p": 0},
        {"n_rep": 0},
        {"base_seed": -1},
        {"dgp_name": "bad_dgp"},
        {"learner_name": "bad_learner"},
        {"theta": 0.0},
    ],
)
def test_scenario_invalid_inputs_raise(kwargs: dict) -> None:
    """Invalid Scenario inputs should raise ValueError."""
    base = {
        "scenario_id": 0,
        "name": "valid",
        "dgp_name": "linear_baseline",
        "learner_name": "lasso",
        "n": 100,
        "p": 150,
        "theta": config.THETA_TRUE,
        "n_rep": TEST_N_REP,
        "base_seed": config.BASE_SEED,
    }
    base.update(kwargs)

    with pytest.raises(ValueError):
        Scenario(**base)


def test_theta_allows_tiny_floating_error() -> None:
    """Theta validation should allow tiny floating-point noise."""
    scenario = Scenario(
        scenario_id=1,
        name="tiny_theta_noise",
        dgp_name="linear_baseline",
        learner_name="ols",
        n=100,
        p=50,
        theta=config.THETA_TRUE + 1e-10,
        n_rep=10,
        base_seed=config.BASE_SEED,
    )
    assert scenario.theta == config.THETA_TRUE + 1e-10


def test_theta_rejects_material_difference() -> None:
    """Theta validation should reject values outside tolerance."""
    with pytest.raises(ValueError):
        Scenario(
            scenario_id=1,
            name="bad_theta",
            dgp_name="linear_baseline",
            learner_name="ols",
            n=100,
            p=50,
            theta=config.THETA_TRUE + 1e-6,
            n_rep=10,
            base_seed=config.BASE_SEED,
        )


def test_build_main_scenarios_returns_list() -> None:
    """Builder should return a list of scenarios."""
    scenarios = build_main_scenarios(n_rep=TEST_N_REP)
    assert isinstance(scenarios, list)


def test_build_main_scenarios_elements_are_scenarios() -> None:
    """All built items should be Scenario instances."""
    scenarios = build_main_scenarios(n_rep=TEST_N_REP)
    assert all(isinstance(item, Scenario) for item in scenarios)


def test_build_main_scenarios_count_is_36() -> None:
    """Main design should produce 36 scenarios."""
    scenarios = build_main_scenarios(n_rep=TEST_N_REP)
    expected_count = (
        len(config.N_VALUES) * len(config.P_VALUES) * len(config.DGP_NAMES) * len(config.LEARNERS)
    )
    assert len(scenarios) == expected_count == 36


def test_main_design_constants_are_fixed() -> None:
    """Main design constants should keep agreed FAST/FULL and folds."""
    assert config.N_FOLDS == 2
    assert config.FAST_N_REP == 500
    assert config.FULL_N_REP == 1000
    assert config.N_VALUES == [200, 300, 400]
    assert config.P_VALUES == [100, 150]
    assert config.DGP_NAMES == ["linear_baseline", "linear_sparse_correlated"]
    assert config.LEARNERS == ["ols", "lasso", "elastic_net"]


def test_build_main_scenarios_ids_are_unique() -> None:
    """Scenario IDs should be unique."""
    scenarios = build_main_scenarios(n_rep=TEST_N_REP)
    ids = [scenario.scenario_id for scenario in scenarios]
    assert len(ids) == len(set(ids))


def test_build_main_scenarios_names_are_unique() -> None:
    """Scenario names should be unique."""
    scenarios = build_main_scenarios(n_rep=TEST_N_REP)
    names = [scenario.name for scenario in scenarios]
    assert len(names) == len(set(names))


def test_first_scenario_has_deterministic_fields() -> None:
    """First built scenario should follow deterministic ordering."""
    first = build_main_scenarios(n_rep=TEST_N_REP)[0]
    expected_name = make_scenario_name(
        dgp_name="linear_baseline",
        learner_name="ols",
        n=config.N_VALUES[0],
        p=config.P_VALUES[0],
    )

    assert first.scenario_id == 0
    assert first.name == expected_name
    assert first.dgp_name == "linear_baseline"
    assert first.learner_name == "ols"
    assert first.n == config.N_VALUES[0]
    assert first.p == config.P_VALUES[0]
    assert first.theta == config.THETA_TRUE
    assert first.n_rep == TEST_N_REP
    assert first.base_seed == config.BASE_SEED


def test_count_main_scenarios_equals_36() -> None:
    """Count helper should match expected design size."""
    expected_count = (
        len(config.N_VALUES) * len(config.P_VALUES) * len(config.DGP_NAMES) * len(config.LEARNERS)
    )
    assert count_main_scenarios() == expected_count == 36


def test_build_main_scenarios_applies_requested_replications() -> None:
    """Builder should apply provided n_rep to every scenario."""
    scenarios = build_main_scenarios(n_rep=100)
    assert all(scenario.n_rep == 100 for scenario in scenarios)


def test_main_scenarios_match_expected_design_sets() -> None:
    """Main scenarios should use the configured n/p x DGP x learner grid."""
    scenarios = build_main_scenarios(n_rep=TEST_N_REP)
    assert {scenario.theta for scenario in scenarios} == {config.THETA_TRUE}
    assert {scenario.n for scenario in scenarios} == set(config.N_VALUES)
    assert {scenario.p for scenario in scenarios} == set(config.P_VALUES)
    assert {scenario.dgp_name for scenario in scenarios} == {
        "linear_baseline",
        "linear_sparse_correlated",
    }
    assert {scenario.learner_name for scenario in scenarios} == {"ols", "lasso", "elastic_net"}
    assert {(scenario.dgp_name, scenario.learner_name) for scenario in scenarios} == {
        ("linear_baseline", "ols"),
        ("linear_baseline", "lasso"),
        ("linear_baseline", "elastic_net"),
        ("linear_sparse_correlated", "ols"),
        ("linear_sparse_correlated", "lasso"),
        ("linear_sparse_correlated", "elastic_net"),
    }


def test_main_scenarios_design_keys_are_unique() -> None:
    """Each main design combination should appear exactly once."""
    scenarios = build_main_scenarios(n_rep=TEST_N_REP)
    keys = [
        (
            scenario.dgp_name,
            scenario.learner_name,
            scenario.n,
            scenario.p,
            scenario.theta,
        )
        for scenario in scenarios
    ]
    expected_count = (
        len(config.N_VALUES) * len(config.P_VALUES) * len(config.DGP_NAMES) * len(config.LEARNERS)
    )
    assert len(keys) == len(set(keys))
    assert len(keys) == expected_count == 36


def _small_test_scenario() -> Scenario:
    return Scenario(
        scenario_id=7,
        name="test_linear_lasso",
        dgp_name="linear_baseline",
        learner_name="lasso",
        n=60,
        p=10,
        theta=config.THETA_TRUE,
        n_rep=3,
        base_seed=config.BASE_SEED,
    )


def test_run_single_replication_returns_required_keys() -> None:
    """Single replication result should include required output keys."""
    scenario = _small_test_scenario()
    row = run_single_replication(scenario=scenario, replication=0)
    required_keys = {
        "scenario_id",
        "scenario_name",
        "replication",
        "dgp_name",
        "learner_name",
        "n",
        "p",
        "theta_true",
        "data_seed",
        "split_seed",
        "learner_g_seed",
        "learner_m_seed",
        "theta_hat",
        "se",
        "t_stat",
        "ci_lower",
        "ci_upper",
        "ci_length",
        "covered",
        "error",
        "squared_error",
    }
    assert required_keys.issubset(set(row.keys()))
    assert row["ci_length"] >= 0


def test_run_scenario_returns_dataframe_with_expected_rows() -> None:
    """Scenario runner should return one row per replication."""
    scenario = _small_test_scenario()
    results = run_scenario(scenario)
    assert isinstance(results, pd.DataFrame)
    assert len(results) == scenario.n_rep


def test_run_scenario_reproducible_for_same_scenario() -> None:
    """Repeated scenario runs should be deterministic with same seeds."""
    scenario = _small_test_scenario()
    first = run_scenario(scenario)
    second = run_scenario(scenario)
    pd.testing.assert_frame_equal(first, second)


def test_get_replication_count_uses_pipeline_mode_values() -> None:
    """Replication count mapping should be owned by pipeline_mode."""
    assert get_replication_count("fast") == config.FAST_N_REP
    assert get_replication_count("full") == config.FULL_N_REP
    assert get_replication_count("FAST") == config.FAST_N_REP
    assert get_replication_count("FULL") == config.FULL_N_REP


def test_get_run_mode_reads_env_and_validates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run mode should default to full and reject unsupported values."""
    monkeypatch.delenv("DML_RUN_MODE", raising=False)
    assert get_run_mode() == "full"

    monkeypatch.setenv("DML_RUN_MODE", "FAST")
    assert get_run_mode() == "fast"

    monkeypatch.setenv("DML_RUN_MODE", "bad")
    with pytest.raises(ValueError):
        get_run_mode()
