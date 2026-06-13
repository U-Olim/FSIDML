"""Scenario model and naming helpers for Monte Carlo experiment design."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any

from dml_project import config


@dataclass(frozen=True)
class Scenario:
    """Immutable configuration for one Monte Carlo scenario."""

    dgp_name: str
    learner_name: str
    n_obs: int
    n_covariates: int
    n_folds: int
    scenario_id: int = 0
    theta: float = config.THETA_TRUE
    n_rep: int = config.FULL_N_REP
    base_seed: int = config.BASE_SEED

    def __post_init__(self) -> None:
        """Validate scenario fields against the project design."""

        if self.scenario_id < 0:
            raise ValueError("scenario_id must be >= 0")
        if not isinstance(self.n_obs, int) or self.n_obs <= 0:
            raise ValueError("n_obs must be a positive integer")
        if not isinstance(self.n_covariates, int) or self.n_covariates <= 0:
            raise ValueError("n_covariates must be a positive integer")
        if not isinstance(self.n_folds, int):
            raise ValueError("n_folds must be an integer")
        if self.n_folds < 2:
            raise ValueError("n_folds must be >= 2")
        if self.n_folds > self.n_obs:
            raise ValueError("n_folds must be <= n_obs")
        if self.n_rep <= 0:
            raise ValueError("n_rep must be > 0")
        if self.base_seed < 0:
            raise ValueError("base_seed must be >= 0")
        if self.dgp_name not in config.DGP_NAMES:
            raise ValueError("dgp_name is not supported")
        if self.learner_name not in config.LEARNERS:
            raise ValueError("learner_name is not supported")
        if not math.isclose(self.theta, config.THETA_TRUE, abs_tol=1e-9):
            raise ValueError(f"theta must equal fixed design value {config.THETA_TRUE}")

    @property
    def name(self) -> str:
        """Return a deterministic, file-safe scenario name."""

        return make_scenario_name(
            dgp_name=self.dgp_name,
            learner_name=self.learner_name,
            n_obs=self.n_obs,
            n_covariates=self.n_covariates,
            n_folds=self.n_folds,
        )

    @property
    def fold_train_size(self) -> float:
        """Return the approximate fold-level nuisance training size."""

        return self.n_obs * (1 - 1 / self.n_folds)

    @property
    def n(self) -> int:
        """Backward-compatible alias for n_obs."""

        return self.n_obs

    @property
    def p(self) -> int:
        """Backward-compatible alias for n_covariates."""

        return self.n_covariates

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation of the scenario."""

        data = {field.name: getattr(self, field.name) for field in fields(self)}
        data["name"] = self.name
        data["n"] = self.n
        data["p"] = self.p
        data["fold_train_size"] = self.fold_train_size
        return data

    def short_label(self) -> str:
        """Return a compact human-readable label for logs and tables."""

        return (
            f"{self.dgp_name}_{self.learner_name}_"
            f"n{self.n_obs}_p{self.n_covariates}_k{self.n_folds}_th{self.theta}"
        )


def make_scenario_name(
    dgp_name: str,
    learner_name: str,
    n_obs: int,
    n_covariates: int,
    n_folds: int,
) -> str:
    """Create a deterministic, file-safe scenario name from key fields."""

    return f"{dgp_name}_n{n_obs}_p{n_covariates}_k{n_folds}_{learner_name}"
