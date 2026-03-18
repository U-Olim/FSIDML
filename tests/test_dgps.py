"""Tests for DGP modules and generated data behavior."""

from __future__ import annotations

import numpy as np
import pytest

from dml_project.dgps.base import generate_covariates
from dml_project.dgps import linear_baseline, linear_sparse_correlated


def _decay_coef() -> np.ndarray:
    idx = np.arange(1, 6, dtype=float)
    coef = 1.0 / (idx**2)
    coef = coef / np.linalg.norm(coef)
    return coef


@pytest.mark.parametrize(
    "generator",
    [
        linear_baseline.generate_data,
        linear_sparse_correlated.generate_data,
    ],
)
def test_generate_data_shapes(generator) -> None:
    """DGPs should return arrays with expected shapes."""
    n = 200
    p = 10
    y, d, x = generator(n=n, p=p, theta=2.0, seed=123)
    assert isinstance(y, np.ndarray)
    assert isinstance(d, np.ndarray)
    assert isinstance(x, np.ndarray)
    assert y.shape == (n,)
    assert d.shape == (n,)
    assert x.shape == (n, p)


@pytest.mark.parametrize(
    "generator",
    [
        linear_baseline.generate_data,
        linear_sparse_correlated.generate_data,
    ],
)
def test_generate_data_reproducible_same_seed(generator) -> None:
    """Same seed should generate identical data."""
    first = generator(n=200, p=10, theta=2.0, seed=321)
    second = generator(n=200, p=10, theta=2.0, seed=321)
    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a, b)


@pytest.mark.parametrize(
    "generator",
    [
        linear_baseline.generate_data,
        linear_sparse_correlated.generate_data,
    ],
)
def test_generate_data_changes_with_different_seed(generator) -> None:
    """Different seeds should produce different generated arrays."""
    first = generator(n=200, p=10, theta=2.0, seed=111)
    second = generator(n=200, p=10, theta=2.0, seed=112)
    assert any(not np.array_equal(a, b) for a, b in zip(first, second, strict=True))


def test_covariate_correlation_is_fixed() -> None:
    """Covariate dependence in X should follow the fixed AR(1) design."""
    n = 10000
    p = 5
    seed = 1234

    x = generate_covariates(n=n, p=p, rng=np.random.default_rng(seed))
    corr = np.corrcoef(x[:, 0], x[:, 1])[0, 1]
    assert 0.45 < corr < 0.55


@pytest.mark.parametrize(
    "generator,recover_m0,recover_g0",
    [
        (
            linear_baseline.generate_data,
            lambda x: x[:, :5] @ _decay_coef(),
            lambda x: 0.5 * (x[:, :5] @ _decay_coef()),
        ),
        (
            linear_sparse_correlated.generate_data,
            lambda x: x[:, :5] @ _decay_coef(),
            lambda x: 0.5 * (x[:, :5] @ _decay_coef()),
        ),
    ],
)
def test_error_correlation_follows_fixed_baseline(
    generator, recover_m0, recover_g0
) -> None:
    """Recovered corr(epsilon, v) should follow the fixed baseline design."""
    theta = 2.0
    y, d, x = generator(n=8000, p=10, theta=theta, seed=999)
    v = d - recover_m0(x)
    epsilon = y - theta * d - recover_g0(x)
    corr = np.corrcoef(epsilon, v)[0, 1]
    assert np.isclose(corr, 0.0, atol=0.03)


def test_linear_sparse_correlated_theta_is_consistent_with_true_plr_equation() -> None:
    """Using true nuisances should recover theta in expectation."""
    theta = 1.0
    estimates = []
    for seed in range(30):
        y, d, x = linear_sparse_correlated.generate_data(n=1500, p=10, theta=theta, seed=seed)
        signal = x[:, :5] @ _decay_coef()
        m0 = signal
        g0 = 0.5 * signal
        y_res = y - g0
        d_res = d - m0
        estimates.append(np.mean(d_res * y_res) / np.mean(d_res**2))

    assert np.isclose(float(np.mean(estimates)), theta, atol=0.03)
