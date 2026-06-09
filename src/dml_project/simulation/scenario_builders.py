"""Builders for deterministic simulation scenario grids."""

from __future__ import annotations

from collections.abc import Iterable

from dml_project import config
from dml_project.simulation.scenario import Scenario


def _build_scenarios(
    n_p_pairs: Iterable[tuple[int, int]],
    n_rep: int,
) -> list[Scenario]:
    """Build scenarios for the provided sample-size and dimension pairs."""

    scenarios: list[Scenario] = []
    scenario_id = 0

    for n_obs, n_covariates in n_p_pairs:
        for n_folds in config.K_VALUES:
            for dgp_name in config.DGP_NAMES:
                for learner_name in config.LEARNERS:
                    scenarios.append(
                        Scenario(
                            scenario_id=scenario_id,
                            dgp_name=dgp_name,
                            learner_name=learner_name,
                            n_obs=n_obs,
                            n_covariates=n_covariates,
                            n_folds=n_folds,
                            theta=config.THETA_TRUE,
                            n_rep=n_rep,
                            base_seed=config.BASE_SEED,
                        )
                    )
                    scenario_id += 1

    return scenarios


def build_all_scenarios(n_rep: int = config.FULL_N_REP) -> list[Scenario]:
    """Build the full revised scenario grid."""

    n_p_pairs = (
        (n_obs, n_covariates)
        for n_obs in config.N_VALUES
        for n_covariates in config.P_VALUES
    )
    scenarios = _build_scenarios(n_p_pairs=n_p_pairs, n_rep=n_rep)
    expected_count = count_all_scenarios()

    if len(scenarios) != expected_count:
        raise ValueError(
            "build_all_scenarios produced unexpected scenario count: "
            f"expected={expected_count}, got={len(scenarios)}."
        )
    return scenarios


def build_pilot_scenarios(n_rep: int = config.FAST_N_REP) -> list[Scenario]:
    """Build the reduced pilot scenario grid."""

    scenarios = _build_scenarios(n_p_pairs=config.PILOT_N_P_PAIRS, n_rep=n_rep)
    expected_count = count_pilot_scenarios()

    if len(scenarios) != expected_count:
        raise ValueError(
            "build_pilot_scenarios produced unexpected scenario count: "
            f"expected={expected_count}, got={len(scenarios)}."
        )
    return scenarios


def build_main_scenarios(n_rep: int) -> list[Scenario]:
    """Backward-compatible alias for the full scenario grid."""

    return build_all_scenarios(n_rep=n_rep)


def count_all_scenarios() -> int:
    """Return the total number of scenarios in the full design grid."""

    return (
        len(config.N_VALUES)
        * len(config.P_VALUES)
        * len(config.K_VALUES)
        * len(config.DGP_NAMES)
        * len(config.LEARNERS)
    )


def count_pilot_scenarios() -> int:
    """Return the total number of scenarios in the pilot design grid."""

    return (
        len(config.PILOT_N_P_PAIRS)
        * len(config.K_VALUES)
        * len(config.DGP_NAMES)
        * len(config.LEARNERS)
    )


def count_main_scenarios() -> int:
    """Backward-compatible alias for the full scenario count."""

    return count_all_scenarios()
