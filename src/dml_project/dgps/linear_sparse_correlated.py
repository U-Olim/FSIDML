"""Correlated approximately sparse linear DGP for PLR simulations.

Compared to the baseline linear DGP, this variant uses an equicorrelated
covariate design to stress nuisance estimators under stronger dependence
among regressors.
"""

from __future__ import annotations

import numpy as np

from dml_project.dgps.base import generate_baseline_errors

_COVARIATE_CORRELATION = 0.5
_N_RELEVANT = 5
_GAMMA_TREATMENT = 1.0
_GAMMA_OUTCOME = 0.5


def _equicorrelation_covariance(p: int, rho: float) -> np.ndarray:
    """Build an equicorrelation covariance matrix.

    Args:
        p: Matrix dimension (number of covariates).
        rho: Off-diagonal correlation shared by all variable pairs.

    Returns:
        A ``(p, p)`` covariance matrix with ones on the diagonal and ``rho``
        off the diagonal.
    """

    return (1.0 - rho) * np.eye(p) + rho * np.ones((p, p))


def _decay_coefficients(s: int) -> np.ndarray:
    """Return normalized sparse-like coefficients with polynomial decay.

    Args:
        s: Number of active (relevant) coefficients.

    Returns:
        Vector of length ``s`` proportional to ``1 / j^2`` and normalized to
        Euclidean norm 1.
    """

    idx = np.arange(1, s + 1, dtype=float)
    coef = 1.0 / (idx**2)
    coef = coef / np.linalg.norm(coef)
    return coef


def generate_data(
    n: int, p: int, theta: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate ``(Y, D, X)`` under correlated covariates and sparse signals.

    Args:
        n: Number of observations.
        p: Number of covariates.
        theta: True causal effect in the PLR outcome equation.
        seed: Seed used to initialize the random number generator.

    Returns:
        Tuple ``(y, d, x)`` where ``y`` and ``d`` are one-dimensional arrays
        of length ``n`` and ``x`` has shape ``(n, p)``.
    """

    if p < 1:
        raise ValueError("p must be >= 1 for linear_sparse_correlated")

    rng = np.random.default_rng(seed)
    covariance = _equicorrelation_covariance(p=p, rho=_COVARIATE_CORRELATION)
    x = rng.multivariate_normal(mean=np.zeros(p), cov=covariance, size=n)
    epsilon, v = generate_baseline_errors(n=n, rng=rng)

    s = min(_N_RELEVANT, p)
    beta = _decay_coefficients(s=s)
    signal = x[:, :s] @ beta

    m0 = _GAMMA_TREATMENT * signal
    g0 = _GAMMA_OUTCOME * signal

    d = m0 + v
    y = theta * d + g0 + epsilon
    return y, d, x
