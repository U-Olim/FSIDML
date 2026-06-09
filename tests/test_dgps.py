"""Tests for Fuhr-style DGP modules and shared helpers."""

from __future__ import annotations

import importlib
from collections.abc import Callable

import numpy as np
import pytest

from dml_project import config
from dml_project.dgps.base import (
    active_dimension,
    centered_quadratic,
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
    assert _theta_0() == 1.0


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


def test_dgp_names_are_fuhr_style_names_only() -> None:
    """Config should not include deprecated DGP names."""

    assert config.DGP_NAMES == [
        "linear_confounding",
        "quadratic_confounding",
        "interaction_confounding",
        "step_confounding",
    ]
    assert "linear_dense_independent" not in config.DGP_NAMES
    assert "linear_sparse_correlated" not in config.DGP_NAMES
    assert "nonlinear_smooth" not in config.DGP_NAMES
    assert "threshold_interaction" not in config.DGP_NAMES
