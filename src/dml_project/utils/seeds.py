"""Utilities for deterministic seed management in Monte Carlo simulations."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class SeedBundle:
    """Container for all seeds used in one scenario replication.

    Attributes:
        base_seed: Global base seed configured for the project.
        scenario_id: Non-negative scenario identifier.
        replication: Non-negative replication index within the scenario.
        data_seed: Deterministic seed used for data generation.
        split_seed: Deterministic seed used for fold assignment/splitting.
        learner_g_seed: Deterministic seed used by nuisance learner g.
        learner_m_seed: Deterministic seed used by nuisance learner m.
    """

    base_seed: int
    scenario_id: int
    replication: int
    data_seed: int
    split_seed: int
    learner_g_seed: int
    learner_m_seed: int


def make_seed_bundle(scenario_id: int, replication: int, base_seed: int) -> SeedBundle:
    """Create deterministic independent seeds for one scenario-replication pair.

    Seed generation is based on ``numpy.random.SeedSequence.spawn`` to produce
    independent child streams for data generation, sample splitting, and each
    nuisance learner.

    Args:
        scenario_id: Scenario identifier (must be >= 0).
        replication: Replication index (must be >= 0).
        base_seed: Base seed value (must be >= 0).

    Returns:
        SeedBundle with deterministic seeds for data, split, learner g, learner m.

    Raises:
        ValueError: If any input is negative.
    """

    if scenario_id < 0:
        raise ValueError("scenario_id must be >= 0")
    if replication < 0:
        raise ValueError("replication must be >= 0")
    if base_seed < 0:
        raise ValueError("base_seed must be >= 0")

    base_ss = np.random.SeedSequence(base_seed)
    scenario_ss = base_ss.spawn(scenario_id + 1)[scenario_id]
    replication_ss = scenario_ss.spawn(replication + 1)[replication]
    children = replication_ss.spawn(4)
    data_seed = int(children[0].generate_state(1, dtype=np.uint32)[0])
    split_seed = int(children[1].generate_state(1, dtype=np.uint32)[0])
    learner_g_seed = int(children[2].generate_state(1, dtype=np.uint32)[0])
    learner_m_seed = int(children[3].generate_state(1, dtype=np.uint32)[0])
    return SeedBundle(
        base_seed=base_seed,
        scenario_id=scenario_id,
        replication=replication,
        data_seed=data_seed,
        split_seed=split_seed,
        learner_g_seed=learner_g_seed,
        learner_m_seed=learner_m_seed,
    )


def make_numpy_rng(seed: int) -> np.random.Generator:
    """Return a NumPy random Generator initialized with the given seed."""

    return np.random.default_rng(seed)


def seed_bundle_to_dict(bundle: SeedBundle) -> dict[str, int]:
    """Convert a SeedBundle to a plain dictionary."""

    return asdict(bundle)
