"""Scenario model and naming helpers for Monte Carlo experiment design."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from dml_project import config

_ALLOWED_DGP_NAMES = {"linear_baseline", "linear_sparse_correlated"}
_ALLOWED_LEARNER_NAMES = {"ols", "lasso", "elastic_net"}


@dataclass(frozen=True)
class Scenario:
    """Immutable configuration for one Monte Carlo scenario.

    Each instance defines a single cell of the design grid (DGP, learner,
    sample size, dimensionality, and replication count).
    """

    scenario_id: int
    name: str
    dgp_name: str
    learner_name: str
    n: int
    p: int
    theta: float
    n_rep: int
    base_seed: int
    matched_specification: bool = False

    def __post_init__(self) -> None:
        """Validate scenario fields against the project design."""

        if self.scenario_id < 0:
            raise ValueError("scenario_id must be >= 0")
        if self.n <= 0:
            raise ValueError("n must be > 0")
        if self.p <= 0:
            raise ValueError("p must be > 0")
        if self.n_rep <= 0:
            raise ValueError("n_rep must be > 0")
        if self.base_seed < 0:
            raise ValueError("base_seed must be >= 0")
        if self.dgp_name not in _ALLOWED_DGP_NAMES:
            raise ValueError("dgp_name is not supported")
        if self.learner_name not in _ALLOWED_LEARNER_NAMES:
            raise ValueError("learner_name is not supported")
        if not math.isclose(self.theta, config.THETA_TRUE, abs_tol=1e-9):
            raise ValueError(f"theta must equal fixed design value {config.THETA_TRUE}")
        # All active DGPs in the final design are linear PLR specifications.
        matched_specification = self.dgp_name in _ALLOWED_DGP_NAMES
        object.__setattr__(self, "matched_specification", matched_specification)

    def to_dict(self) -> dict:
        """Return a plain dictionary representation of the scenario.

        Returns:
            Scenario fields as a standard Python dictionary.
        """

        return asdict(self)

    def short_label(self) -> str:
        """Return a compact human-readable label for logs and tables.

        Returns:
            Deterministic short label combining key scenario fields.
        """

        return (
            f"{self.dgp_name}_{self.learner_name}_"
            f"n{self.n}_p{self.p}_th{self.theta}"
        )


def make_scenario_name(
    dgp_name: str,
    learner_name: str,
    n: int,
    p: int,
) -> str:
    """Create a deterministic, file-safe scenario name from key fields.

    Args:
        dgp_name: DGP module identifier.
        learner_name: Learner identifier.
        n: Sample size.
        p: Number of covariates.

    Returns:
        String name suitable for logs and file naming.
    """

    return f"{dgp_name}_n{n}_p{p}_{learner_name}"
