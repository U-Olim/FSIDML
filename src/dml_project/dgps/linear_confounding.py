"""Fuhr-style linear confounding DGP for PLR simulations."""

from __future__ import annotations

import numpy as np

from dml_project.dgps.base import (
    active_dimension,
    coefficient_vectors,
    generate_independent_covariates,
    make_plr_data,
)


def generate_data(
    n: int,
    p: int,
    theta: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate ``(Y, D, X)`` with linear nuisance functions."""

    rng = np.random.default_rng(seed)
    x = generate_independent_covariates(n_obs=n, n_covariates=p, rng=rng)
    s = active_dimension(n_covariates=p)
    a, b = coefficient_vectors(n_active=s)
    x_active = x[:, :s]
    r0 = x_active @ a
    g0 = x_active @ b
    return make_plr_data(x=x, r0=r0, g0=g0, theta_0=theta, rng=rng)
