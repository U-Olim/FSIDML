"""Tests for linear PLR DGP modules and shared helpers."""

from __future__ import annotations

import importlib
from collections.abc import Callable

import numpy as np
import pytest

from dml_project import config
from dml_project.dgps.base import (
    active_dimension,
    ar1_covariance,
    centered_quadratic,
    sparse_coefficients,
    standard_normal_step,
)

DGPOutput = tuple[np.ndarray, np.ndarray, np.ndarray]
DGPGenerator = Callable[[int, int, float, int], DGPOutput]


def _theta_0() -> float:
    return getattr(config, "THETA_0", 1.0)


def _generator(dgp_name: str) -> DGPGenerator:
    module = importlib.import_module(f"dml_project.dgps.{dgp_name}")
    return module.generate_data


@pytest.mark.parametrize("dgp_name", config.DGP_NAMES)
def test_every_configured_dgp_generates_expected_shapes(dgp_name: str) -> None:
    """Every configured DGP should generate the runner-compatible tuple."""

    n_obs = 50
    n_covariates = 20
    y, d, x = _generator(dgp_name)(
        n_obs,
        n_covariates,
        _theta_0(),
        123,
    )

    assert x.shape == (n_obs, n_covariates)
    assert d.shape == (n_obs,)
    assert y.shape == (n_obs,)
    assert _theta_0() == config.THETA_0


@pytest.mark.parametrize("dgp_name", config.DGP_NAMES)
def test_generate_data_reproducible_same_seed(dgp_name: str) -> None:
    """Same seed should generate identical arrays."""

    first = _generator(dgp_name)(50, 20, _theta_0(), 321)
    second = _generator(dgp_name)(50, 20, _theta_0(), 321)

    for first_array, second_array in zip(first, second, strict=True):
        assert np.array_equal(first_array, second_array)


@pytest.mark.parametrize("dgp_name", config.DGP_NAMES)
def test_generate_data_changes_with_different_seed(dgp_name: str) -> None:
    """Different seeds should change generated data."""

    first = _generator(dgp_name)(50, 20, _theta_0(), 111)
    second = _generator(dgp_name)(50, 20, _theta_0(), 112)

    assert any(
        not np.array_equal(first_array, second_array)
        for first_array, second_array in zip(first, second, strict=True)
    )


def test_active_dimension_caps_at_ten() -> None:
    """Active dimension should be min(10, p)."""

    assert active_dimension(5) == 5
    assert active_dimension(20) == 10


def test_standard_normal_step_values() -> None:
    """Step helper should implement the fixed quartile thresholds."""

    actual = standard_normal_step(np.array([-1.0, -0.1, 0.1, 1.0]))

    assert np.array_equal(actual, np.array([-3.0, -1.0, 1.0, 3.0]))


def test_centered_quadratic_values() -> None:
    """Quadratic helper should return x squared minus one."""

    actual = centered_quadratic(np.array([0.0, 1.0, 2.0]))

    assert np.array_equal(actual, np.array([-1.0, 0.0, 3.0]))


def test_ar1_covariance_values() -> None:
    """AR(1) covariance should use rho to the distance power."""

    expected = np.array(
        [
            [1.0, 0.5, 0.25, 0.125],
            [0.5, 1.0, 0.5, 0.25],
            [0.25, 0.5, 1.0, 0.5],
            [0.125, 0.25, 0.5, 1.0],
        ],
    )

    assert np.array_equal(ar1_covariance(4, 0.5), expected)


@pytest.mark.parametrize(
    ("n_covariates", "expected_nonzero"),
    [(20, 10), (5, 5)],
)
def test_sparse_coefficients_active_count(
    n_covariates: int,
    expected_nonzero: int,
) -> None:
    """Sparse helper should cap active coefficients at p."""

    coefficients = sparse_coefficients(n_covariates, n_active=10)

    assert np.count_nonzero(coefficients) == expected_nonzero


def test_weak_signal_sparse_has_smaller_signal_norm() -> None:
    """Weak-signal coefficients should be smaller than baseline sparse ones."""

    baseline = sparse_coefficients(20, n_active=10, scale=1.0)
    weak = sparse_coefficients(20, n_active=10, scale=0.25)

    assert np.linalg.norm(weak) < np.linalg.norm(baseline)


def test_dgp_names_are_linear_plr_names_only() -> None:
    """Config should include only the active linear PLR DGP names."""

    assert config.DGP_NAMES == [
        "dense_linear_independent",
        "sparse_linear_independent",
        "sparse_linear_correlated",
        "weak_signal_sparse",
    ]
