"""Shared helpers for generating DGP inputs in simulation experiments.

This module centralizes reusable random-data primitives used by multiple
data-generating processes (DGPs), so scenario modules can focus only on their
structural equations.
"""

from __future__ import annotations

import numpy as np

_X_CORR = 0.5
_ERROR_CORR = 0.0


def generate_covariates(n: int, p: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a Gaussian covariate matrix with AR(1)-style dependence.

    Args:
        n: Number of observations.
        p: Number of covariates.
        rng: NumPy random number generator used for reproducibility.

    Returns:
        Array with shape ``(n, p)`` sampled from a zero-mean multivariate
        normal distribution whose covariance is
        ``Cov(X_j, X_k) = _X_CORR ** abs(j-k)``.
    """

    if n <= 0:
        raise ValueError("n must be > 0")
    if p <= 0:
        raise ValueError("p must be > 0")

    indices = np.arange(p)
    covariance = _X_CORR ** np.abs(np.subtract.outer(indices, indices))
    mean = np.zeros(p)
    return rng.multivariate_normal(mean=mean, cov=covariance, size=n)


def generate_baseline_errors(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate baseline Gaussian disturbances for outcome and treatment.

    Args:
        n: Number of observations.
        rng: NumPy random number generator used for reproducibility.

    Returns:
        A tuple ``(epsilon, v)`` where each element has shape ``(n,)`` and is
        sampled from a bivariate normal distribution with unit variances and
        correlation ``_ERROR_CORR``.
    """

    if n <= 0:
        raise ValueError("n must be > 0")
    covariance = np.array([[1.0, _ERROR_CORR], [_ERROR_CORR, 1.0]])
    errors = rng.multivariate_normal(mean=np.zeros(2), cov=covariance, size=n)
    epsilon = errors[:, 0]
    v = errors[:, 1]
    return epsilon, v
