"""Shared helpers for PLR data-generating processes."""

from __future__ import annotations

import numpy as np

_STEP_CUTOFF = 0.67448975


def generate_independent_covariates(
    n_obs: int,
    n_covariates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate independent standard-normal covariates."""

    if n_obs <= 0:
        raise ValueError("n_obs must be > 0")
    if n_covariates <= 0:
        raise ValueError("n_covariates must be > 0")
    return rng.normal(loc=0.0, scale=1.0, size=(n_obs, n_covariates))


def active_dimension(n_covariates: int, max_active: int = 10) -> int:
    """Return the number of active covariates."""

    if n_covariates <= 0:
        raise ValueError("n_covariates must be > 0")
    if max_active <= 0:
        raise ValueError("max_active must be > 0")
    return min(max_active, n_covariates)


def coefficient_vectors(n_active: int) -> tuple[np.ndarray, np.ndarray]:
    """Return treatment and outcome nuisance coefficient vectors."""

    if n_active <= 0:
        raise ValueError("n_active must be > 0")
    j = np.arange(1, n_active + 1, dtype=float)
    a = 1.0 / j
    b = ((-1.0) ** (j + 1.0)) / j
    return a, b


def centered_quadratic(x: np.ndarray) -> np.ndarray:
    """Return centered quadratic terms."""

    return x**2 - 1.0


def standard_normal_step(x: np.ndarray) -> np.ndarray:
    """Apply a fixed four-level standard-normal quartile step function."""

    return np.select(
        [
            x < -_STEP_CUTOFF,
            (x >= -_STEP_CUTOFF) & (x < 0.0),
            (x >= 0.0) & (x < _STEP_CUTOFF),
            x >= _STEP_CUTOFF,
        ],
        [-3.0, -1.0, 1.0, 3.0],
    )


def make_plr_data(
    x: np.ndarray,
    r0: np.ndarray,
    g0: np.ndarray,
    theta_0: float,
    rng: np.random.Generator,
    sigma_u: float = 1.0,
    sigma_v: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate ``(Y, D, X)`` from the partially linear model."""

    n_obs = x.shape[0]
    u = rng.normal(loc=0.0, scale=sigma_u, size=n_obs)
    v = rng.normal(loc=0.0, scale=sigma_v, size=n_obs)
    d = r0 + v
    y = theta_0 * d + g0 + u
    return y, d, x


def generate_covariates(n: int, p: int, rng: np.random.Generator) -> np.ndarray:
    """Backward-compatible alias for independent covariate generation."""

    return generate_independent_covariates(
        n_obs=n,
        n_covariates=p,
        rng=rng,
    )


def generate_baseline_errors(
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate independent Gaussian outcome and treatment errors."""

    if n <= 0:
        raise ValueError("n must be > 0")
    epsilon = rng.normal(loc=0.0, scale=1.0, size=n)
    v = rng.normal(loc=0.0, scale=1.0, size=n)
    return epsilon, v
