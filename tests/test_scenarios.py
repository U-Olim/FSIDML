"""Tests for simulation scenario datatypes and builders."""

from __future__ import annotations

import pytest

from dml_project import config
from dml_project.simulation.scenario import Scenario, make_scenario_name
from dml_project.simulation.scenario_builders import (
    build_all_scenarios,
    build_main_scenarios,
    build_pilot_scenarios,
    count_all_scenarios,
    count_main_scenarios,
    count_pilot_scenarios,
)


def _valid_scenario(**overrides: object) -> Scenario:
    values = {
        "scenario_id": 0,
        "dgp_name": "linear_confounding",
        "learner_name": "lasso",
        "n_obs": 500,
        "n_covariates": 50,
        "n_folds": 5,
        "theta": config.THETA_TRUE,
        "n_rep": 10,
        "base_seed": config.BASE_SEED,
    }
    values.update(overrides)
    return Scenario(**values)


def test_config_contains_revised_design_constants() -> None:
    """Config should expose the revised scenario universe."""

    assert config.N_VALUES == [250, 500, 1000]
    assert config.P_VALUES == [25, 50, 100, 150, 300]
    assert config.K_VALUES == [2, 5, 10]
    assert config.DEFAULT_N_FOLDS == 2
    assert config.N_FOLDS == config.DEFAULT_N_FOLDS
    assert config.INNER_CV_FOLDS == 5
    assert config.PILOT_N_P_PAIRS == [
        (500, 50),
        (500, 150),
        (250, 150),
        (250, 300),
    ]
    assert config.DGP_NAMES == [
        "linear_confounding",
        "quadratic_confounding",
        "interaction_confounding",
        "step_confounding",
    ]
    assert config.LEARNERS == [
        "ols",
        "lasso",
        "elastic_net",
        "random_forest",
        "gradient_boosting",
    ]


@pytest.mark.parametrize("dgp_name", config.DGP_NAMES)
@pytest.mark.parametrize("learner_name", config.LEARNERS)
def test_scenario_accepts_configured_dgps_and_learners(
    dgp_name: str,
    learner_name: str,
) -> None:
    """Scenario validation should use configured DGP and learner names."""

    scenario = _valid_scenario(dgp_name=dgp_name, learner_name=learner_name)

    assert scenario.dgp_name == dgp_name
    assert scenario.learner_name == learner_name


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_obs": 0},
        {"n_covariates": 0},
        {"n_folds": 1},
        {"n_folds": 501},
        {"dgp_name": "unknown_dgp"},
        {"learner_name": "unknown_learner"},
    ],
)
def test_scenario_invalid_inputs_raise(overrides: dict[str, object]) -> None:
    """Invalid scenario fields should raise ValueError."""

    with pytest.raises(ValueError):
        _valid_scenario(**overrides)


def test_scenario_name_includes_fold_marker() -> None:
    """Scenario names should include the cross-fitting fold count."""

    scenario = _valid_scenario(n_folds=5)

    assert "_k5_" in scenario.name
    assert (
        scenario.name
        == "linear_confounding_n500_p50_k5_lasso"
    )


def test_make_scenario_name_includes_fold_marker() -> None:
    """Name helper should be fold-aware."""

    assert (
        make_scenario_name(
            dgp_name="linear_confounding",
            learner_name="lasso",
            n_obs=500,
            n_covariates=50,
            n_folds=5,
        )
        == "linear_confounding_n500_p50_k5_lasso"
    )


def test_scenario_fold_train_size_is_float() -> None:
    """Fold training size should keep the exact approximate value."""

    scenario = _valid_scenario(n_obs=250, n_folds=10)

    assert scenario.fold_train_size == 225.0


def test_scenario_keeps_n_and_p_aliases() -> None:
    """Existing downstream code can still read scenario.n and scenario.p."""

    scenario = _valid_scenario(n_obs=500, n_covariates=150)

    assert scenario.n == 500
    assert scenario.p == 150
    assert scenario.to_dict()["name"] == scenario.name
    assert scenario.to_dict()["fold_train_size"] == scenario.fold_train_size


def test_build_pilot_scenarios_count_matches_design() -> None:
    """Pilot scenarios should use selected n/p pairs over all K/DGP/learner cells."""

    scenarios = build_pilot_scenarios()
    expected_count = (
        len(config.PILOT_N_P_PAIRS)
        * len(config.K_VALUES)
        * len(config.DGP_NAMES)
        * len(config.LEARNERS)
    )

    assert len(scenarios) == expected_count
    assert count_pilot_scenarios() == expected_count


def test_build_all_scenarios_count_matches_design() -> None:
    """Full scenarios should use N x P x K x DGP x learner."""

    scenarios = build_all_scenarios()
    expected_count = (
        len(config.N_VALUES)
        * len(config.P_VALUES)
        * len(config.K_VALUES)
        * len(config.DGP_NAMES)
        * len(config.LEARNERS)
    )

    assert len(scenarios) == expected_count
    assert count_all_scenarios() == expected_count
    assert count_main_scenarios() == expected_count


def test_build_main_scenarios_is_backward_compatible_alias() -> None:
    """Existing callers should get the full scenario grid."""

    scenarios = build_main_scenarios(n_rep=123)

    assert len(scenarios) == count_all_scenarios()
    assert all(scenario.n_rep == 123 for scenario in scenarios)


def test_generated_scenarios_include_k10() -> None:
    """Generated grids should include ten-fold cross-fitting scenarios."""

    scenarios = build_all_scenarios()

    assert any(scenario.n_folds == 10 for scenario in scenarios)


@pytest.mark.parametrize(
    "builder",
    [build_all_scenarios, build_pilot_scenarios],
)
def test_scenario_names_are_unique(builder: object) -> None:
    """Scenario names should uniquely identify design cells."""

    scenarios = builder()
    names = [scenario.name for scenario in scenarios]

    assert len(names) == len(set(names))


def test_generated_scenarios_cover_configured_sets() -> None:
    """Full grid should cover all configured design dimensions."""

    scenarios = build_all_scenarios()

    assert {scenario.n_obs for scenario in scenarios} == set(config.N_VALUES)
    assert {scenario.n_covariates for scenario in scenarios} == set(config.P_VALUES)
    assert {scenario.n_folds for scenario in scenarios} == set(config.K_VALUES)
    assert {scenario.dgp_name for scenario in scenarios} == set(config.DGP_NAMES)
    assert {scenario.learner_name for scenario in scenarios} == set(config.LEARNERS)
