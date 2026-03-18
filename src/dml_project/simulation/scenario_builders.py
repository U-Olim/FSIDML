"""Builders for deterministic simulation scenario grids."""

from __future__ import annotations

from dml_project import config
from dml_project.simulation.scenario import Scenario, make_scenario_name


def build_main_scenarios(n_rep: int) -> list[Scenario]:
    """Build the full main-study scenario grid.

    Args:
        n_rep: Number of Monte Carlo replications per scenario.

    Returns:
        List of :class:`Scenario` objects covering the Cartesian product of
        configured ``n``, ``p``, DGP, and learner values.

    Raises:
        ValueError: If the constructed grid size does not match expectation.
    """

    scenarios: list[Scenario] = []
    scenario_id = 0
    expected_count = (
        len(config.N_VALUES)
        * len(config.P_VALUES)
        * len(config.DGP_NAMES)
        * len(config.LEARNERS)
    )

    for n in config.N_VALUES:
        for p in config.P_VALUES:
            for dgp_name in config.DGP_NAMES:
                for learner_name in config.LEARNERS:
                    name = make_scenario_name(
                        dgp_name=dgp_name,
                        learner_name=learner_name,
                        n=n,
                        p=p,
                    )
                    scenarios.append(
                        Scenario(
                            scenario_id=scenario_id,
                            name=name,
                            dgp_name=dgp_name,
                            learner_name=learner_name,
                            n=n,
                            p=p,
                            theta=config.THETA_TRUE,
                            n_rep=n_rep,
                            base_seed=config.BASE_SEED,
                        )
                    )
                    scenario_id += 1

    actual_count = len(scenarios)
    if actual_count != expected_count:
        raise ValueError(
            "build_main_scenarios produced unexpected scenario count: "
            f"expected={expected_count}, got={actual_count}."
        )
    return scenarios


def count_main_scenarios() -> int:
    """Return the total number of scenarios in the main design grid."""

    return (
        len(config.N_VALUES)
        * len(config.P_VALUES)
        * len(config.DGP_NAMES)
        * len(config.LEARNERS)
    )
