"""Tests for the DML PLR estimator."""

from __future__ import annotations

import numpy as np
import pytest

from dml_project import config
from dml_project.estimators.dml_plr import DMLPLR
from dml_project.learners.elastic_net import ElasticNetCVLearner
from dml_project.learners.lasso import LassoCVLearner
from dml_project.learners.ols import OLSLearner


def _make_plr_data(
    n: int = 300,
    p: int = 6,
    theta: float = 2.0,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    m0 = X[:, 0] + 0.5 * X[:, 1]
    g0 = 0.5 * m0
    v = rng.normal(size=n)
    epsilon = rng.normal(size=n)
    D = m0 + v
    Y = theta * D + g0 + epsilon
    return Y, D, X


def test_dml_plr_runs_without_error() -> None:
    """Estimator should fit on synthetic PLR data."""
    Y, D, X = _make_plr_data()
    est = DMLPLR(
        learner_g=LassoCVLearner(cv=2, random_state=123),
        learner_m=LassoCVLearner(cv=2, random_state=123),
        n_folds=2,
        random_state=123,
    )
    est.fit(Y, D, X)
    assert est.is_fitted_


def test_dml_plr_theta_hat_is_scalar() -> None:
    """theta_hat should be stored as a scalar float."""
    Y, D, X = _make_plr_data()
    est = DMLPLR(
        learner_g=LassoCVLearner(cv=2, random_state=123),
        learner_m=LassoCVLearner(cv=2, random_state=123),
        random_state=123,
    )
    est.fit(Y, D, X)
    assert np.isscalar(est.theta_hat_)
    assert est.n_folds == config.N_FOLDS
    assert est.se_ > 0
    assert est.ci_upper_ > est.ci_lower_


def test_dml_plr_returns_cross_fitted_predictions_and_diagnostics() -> None:
    """Estimator should expose ell/r predictions and fold diagnostics."""

    Y, D, X = _make_plr_data(n=120, p=8, seed=314)
    est = DMLPLR(
        learner_g=OLSLearner(),
        learner_m=OLSLearner(),
        n_folds=3,
        random_state=314,
    )

    est.fit(Y, D, X)

    assert np.isfinite(est.theta_hat)
    assert np.isfinite(est.se)
    assert np.isfinite(est.ci_lower)
    assert np.isfinite(est.ci_upper)
    assert est.ell_hat.shape == (Y.shape[0],)
    assert est.r_hat.shape == (Y.shape[0],)
    assert est.y_residual.shape == (Y.shape[0],)
    assert est.d_residual.shape == (Y.shape[0],)
    assert len(est.fold_diagnostics) == 3

    required_fold_keys = {
        "fold_train_size",
        "fold_test_size",
        "fold_ratio",
        "condition_number",
        "min_eigenvalue",
        "rank_deficient",
        "nuisance_mse_y",
        "nuisance_mse_d",
        "nuisance_r2_y",
        "nuisance_r2_d",
    }
    for fold_diagnostic in est.fold_diagnostics:
        assert required_fold_keys <= set(fold_diagnostic)

    required_aggregate_keys = {
        "mean_fold_train_size",
        "mean_fold_ratio",
        "max_fold_ratio",
        "rank_deficiency_rate",
        "mean_nuisance_mse_y",
        "mean_nuisance_mse_d",
        "mean_nuisance_r2_y",
        "mean_nuisance_r2_d",
    }
    assert required_aggregate_keys <= set(est.aggregate_diagnostics)


def test_dml_plr_respects_requested_n_folds() -> None:
    """Estimator should use the requested number of cross-fitting folds."""

    Y, D, X = _make_plr_data(n=125, p=6, seed=515)
    est = DMLPLR(
        learner_g=OLSLearner(),
        learner_m=OLSLearner(),
        n_folds=5,
        random_state=515,
    )

    est.fit(Y, D, X)

    assert len(est.fold_diagnostics) == 5


def test_dml_plr_keeps_g_m_aliases_for_backward_compatibility() -> None:
    """Deprecated g/m nuisance aliases should match ell/r arrays."""

    Y, D, X = _make_plr_data(n=120, p=6, seed=616)
    est = DMLPLR(
        learner_g=OLSLearner(),
        learner_m=OLSLearner(),
        n_folds=2,
        random_state=616,
    )

    est.fit(Y, D, X)

    assert np.array_equal(est.g_hat_, est.ell_hat_)
    assert np.array_equal(est.m_hat_, est.r_hat_)
    assert np.array_equal(est.g_hat, est.ell_hat)
    assert np.array_equal(est.m_hat, est.r_hat)


def test_dml_plr_works_with_lasso() -> None:
    """Estimator should run with Lasso nuisance learners."""
    Y, D, X = _make_plr_data()
    est = DMLPLR(
        learner_g=LassoCVLearner(cv=2, random_state=7),
        learner_m=LassoCVLearner(cv=2, random_state=7),
        n_folds=2,
        random_state=7,
    )
    est.fit(Y, D, X)
    assert isinstance(est.predict_effect(), float)


def test_dml_plr_works_with_ols() -> None:
    """Estimator should run with OLS nuisance learners."""
    Y, D, X = _make_plr_data()
    est = DMLPLR(
        learner_g=OLSLearner(),
        learner_m=OLSLearner(),
        n_folds=2,
        random_state=7,
    )
    est.fit(Y, D, X)
    assert isinstance(est.predict_effect(), float)


def test_dml_plr_works_with_elastic_net() -> None:
    """Estimator should run with ElasticNet nuisance learners."""
    Y, D, X = _make_plr_data()
    est = DMLPLR(
        learner_g=ElasticNetCVLearner(cv=2, random_state=7),
        learner_m=ElasticNetCVLearner(cv=2, random_state=7),
        n_folds=2,
        random_state=7,
    )
    est.fit(Y, D, X)
    assert isinstance(est.predict_effect(), float)


def test_dml_plr_reproducible_with_fixed_random_state() -> None:
    """Fixed random_state should produce identical theta_hat."""
    Y, D, X = _make_plr_data()

    est1 = DMLPLR(
        learner_g=LassoCVLearner(cv=2, random_state=99),
        learner_m=LassoCVLearner(cv=2, random_state=99),
        n_folds=2,
        random_state=99,
    )
    est2 = DMLPLR(
        learner_g=LassoCVLearner(cv=2, random_state=99),
        learner_m=LassoCVLearner(cv=2, random_state=99),
        n_folds=2,
        random_state=99,
    )

    est1.fit(Y, D, X)
    est2.fit(Y, D, X)

    assert np.isclose(est1.theta_hat_, est2.theta_hat_)
    assert np.allclose(est1.ell_hat_, est2.ell_hat_)
    assert np.allclose(est1.r_hat_, est2.r_hat_)


def test_dml_plr_cross_fitting_has_no_leakage() -> None:
    """Each fold should have disjoint train/valid sets and OOF predictions once."""
    Y, D, X = _make_plr_data(n=120, p=8, seed=888)
    est = DMLPLR(
        learner_g=LassoCVLearner(cv=2, random_state=11),
        learner_m=LassoCVLearner(cv=2, random_state=11),
        n_folds=2,
        random_state=11,
    )
    est.fit(Y, D, X)

    assert np.array_equal(est.oof_counts_, np.ones(X.shape[0], dtype=int))

    covered_valid = np.zeros(X.shape[0], dtype=int)
    for train_idx, valid_idx in est.fold_indices_:
        assert len(set(train_idx).intersection(set(valid_idx))) == 0
        covered_valid[valid_idx] += 1

    assert np.array_equal(covered_valid, np.ones(X.shape[0], dtype=int))


def test_dml_plr_uses_cross_fitted_nuisance_predictions_in_score() -> None:
    """theta_hat/se should match direct score calculations from stored OOF predictions."""
    Y, D, X = _make_plr_data(n=180, p=10, seed=314)
    est = DMLPLR(
        learner_g=LassoCVLearner(cv=2, random_state=21),
        learner_m=LassoCVLearner(cv=2, random_state=21),
        n_folds=2,
        random_state=21,
    )
    est.fit(Y, D, X)

    y_res = est.y_residual_
    d_res = est.d_residual_
    denominator = np.sum(d_res**2)
    theta_manual = float(np.sum(d_res * y_res) / denominator)
    psi = d_res * (y_res - theta_manual * d_res)
    j_manual = float(np.mean(d_res**2))
    var_manual = float(np.mean(psi**2) / (j_manual**2))
    se_manual = float(np.sqrt(var_manual / Y.shape[0]))

    assert np.isclose(est.theta_hat_, theta_manual)
    assert np.isclose(est.se_, se_manual)


def test_dml_plr_near_zero_residualized_treatment_variance_raises() -> None:
    """Estimator should reject unidentified PLR designs."""

    rng = np.random.default_rng(777)
    X = rng.normal(size=(80, 4))
    D = np.ones(80)
    Y = rng.normal(size=80)
    est = DMLPLR(
        learner_g=OLSLearner(),
        learner_m=OLSLearner(),
        n_folds=4,
        random_state=777,
    )

    with pytest.raises(ValueError, match="Near-zero residualized treatment variance"):
        est.fit(Y, D, X)

