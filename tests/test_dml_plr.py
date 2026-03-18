"""Tests for the DML PLR estimator."""

from __future__ import annotations

import numpy as np

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

    y_res = Y - est.g_hat_
    d_res = D - est.m_hat_
    denominator = np.mean(d_res**2)
    theta_manual = float(np.mean(d_res * y_res) / denominator)
    psi = d_res * (y_res - theta_manual * d_res)
    sigma2_manual = float(np.mean(psi**2) / (denominator**2))
    se_manual = float(np.sqrt(sigma2_manual / Y.shape[0]))

    assert np.isclose(est.theta_hat_, theta_manual)
    assert np.isclose(est.se_, se_manual)

