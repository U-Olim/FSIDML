"""Sparse linear PLR DGP with weak nuisance signals."""

from __future__ import annotations

import numpy as np

from dml_project.dgps.base import (
    alternating_sparse_coefficients,
    generate_independent_covariates,
    make_plr_data,
    sparse_coefficients,
)

_SIGNAL_SCALE = 0.25


def generate_data(
    n: int,
    p: int,
    theta: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate ``(Y, D, X)`` with weak sparse linear nuisance functions."""

    rng = np.random.default_rng(seed)
    x = generate_independent_covariates(n_obs=n, n_covariates=p, rng=rng)
    beta_r = sparse_coefficients(n_covariates=p, n_active=10, scale=_SIGNAL_SCALE)
    beta_g = alternating_sparse_coefficients(
        n_covariates=p,
        n_active=10,
        scale=_SIGNAL_SCALE,
    )
    r0 = x @ beta_r
    g0 = x @ beta_g
    return make_plr_data(x=x, r0=r0, g0=g0, theta_0=theta, rng=rng)
