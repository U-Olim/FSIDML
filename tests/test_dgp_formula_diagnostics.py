"""Tests for active DGP formula diagnostics."""

from __future__ import annotations

import numpy as np

from dml_project.utils.dgp_formula_diagnostics import (
    coefficient_vectors_for_dgp,
    covariance_matrix_for_dgp,
    formula_audit_table,
    linear_variance,
)


def test_dense_coefficient_norm_increases_with_p() -> None:
    """Dense harmonic coefficients should grow monotonically with p."""

    audit = formula_audit_table([5, 10, 20])
    dense = audit.loc[audit["dgp_name"] == "dense_linear_independent"]

    assert dense["beta_r_norm"].is_monotonic_increasing
    assert dense["active_dimension"].tolist() == [5, 10, 20]


def test_sparse_coefficient_norm_stabilizes_after_ten() -> None:
    """Sparse coefficients should not change once p exceeds active dimension."""

    beta_10, _ = coefficient_vectors_for_dgp("sparse_linear_independent", 10)
    beta_20, _ = coefficient_vectors_for_dgp("sparse_linear_independent", 20)

    assert np.isclose(np.linalg.norm(beta_10), np.linalg.norm(beta_20))
    assert np.count_nonzero(beta_20) == 10


def test_weak_signal_norm_is_quarter_sparse_norm() -> None:
    """Weak-signal sparse coefficients should be exactly one quarter as large."""

    sparse_beta, _ = coefficient_vectors_for_dgp("sparse_linear_independent", 20)
    weak_beta, _ = coefficient_vectors_for_dgp("weak_signal_sparse", 20)

    assert np.isclose(np.linalg.norm(weak_beta), 0.25 * np.linalg.norm(sparse_beta))


def test_ar1_variance_uses_beta_sigma_beta() -> None:
    """Correlated DGP variance should use beta' Sigma beta."""

    beta, _ = coefficient_vectors_for_dgp("sparse_linear_correlated", 5)
    covariance = covariance_matrix_for_dgp("sparse_linear_correlated", 5)
    expected = float(beta.T @ covariance @ beta)

    assert np.isclose(linear_variance(beta, covariance), expected)
    assert not np.isclose(linear_variance(beta, covariance), float(beta.T @ beta))
