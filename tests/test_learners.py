"""Tests for nuisance learner wrappers and factories."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    from dml_project import config
    from dml_project.learners.elastic_net import ElasticNetCVLearner
    from dml_project.learners.lasso import LassoCVLearner
    from dml_project.learners.ols import OLSLearner
    from dml_project.learners.tuning import make_main_learner
except ModuleNotFoundError:
    from src.dml_project import config
    from src.dml_project.learners.elastic_net import ElasticNetCVLearner
    from src.dml_project.learners.lasso import LassoCVLearner
    from src.dml_project.learners.ols import OLSLearner
    from src.dml_project.learners.tuning import make_main_learner


def _toy_data(n: int = 40, p: int = 6) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    X = rng.normal(size=(n, p))
    beta = np.array([1.0, -0.5, 0.0, 0.3, 0.0, 0.2])
    y = X @ beta + rng.normal(scale=0.1, size=n)
    return X, y


def test_lasso_cv_learner_fit_predict_shape() -> None:
    """LassoCVLearner should fit and return 1d predictions."""
    X, y = _toy_data()
    learner = LassoCVLearner(cv=2, random_state=123)
    learner.fit(X, y)
    preds = learner.predict(X)
    assert preds.shape == (X.shape[0],)


def test_ols_learner_fit_predict_shape() -> None:
    """OLSLearner should fit and return 1d predictions."""
    X, y = _toy_data()
    learner = OLSLearner()
    learner.fit(X, y)
    preds = learner.predict(X)
    assert preds.shape == (X.shape[0],)


def test_elastic_net_cv_learner_fit_predict_shape() -> None:
    """ElasticNetCVLearner should fit and return 1d predictions."""
    X, y = _toy_data()
    learner = ElasticNetCVLearner(cv=2, random_state=123)
    learner.fit(X, y)
    preds = learner.predict(X)
    assert preds.shape == (X.shape[0],)


@pytest.mark.parametrize(
    "learner",
    [OLSLearner(), LassoCVLearner(), ElasticNetCVLearner(random_state=123)],
)
def test_predict_before_fit_raises_runtime_error(learner) -> None:
    """Predict before fit should raise RuntimeError."""
    X, _ = _toy_data()
    with pytest.raises(RuntimeError):
        learner.predict(X)


def test_make_main_learner_lasso_type() -> None:
    """Factory should return LassoCVLearner for lasso."""
    learner = make_main_learner("lasso")
    assert isinstance(learner, LassoCVLearner)


def test_make_main_learner_ols_type() -> None:
    """Factory should return OLSLearner for ols."""
    learner = make_main_learner("ols")
    assert isinstance(learner, OLSLearner)


def test_make_main_learner_elastic_net_type() -> None:
    """Factory should return ElasticNetCVLearner for elastic_net."""
    learner = make_main_learner("elastic_net")
    assert isinstance(learner, ElasticNetCVLearner)


def test_factory_uses_project_cv_for_penalized_learners() -> None:
    """Factory should use shared CV convention across penalized learners."""
    lasso = make_main_learner("lasso")
    elastic = make_main_learner("elastic_net")
    assert isinstance(lasso, LassoCVLearner)
    assert lasso.cv == config.N_FOLDS
    assert isinstance(elastic, ElasticNetCVLearner)
    assert elastic.cv == config.N_FOLDS


@pytest.mark.parametrize("bad_name", ["ridge", ""])
def test_invalid_learner_name_raises_value_error(bad_name: str) -> None:
    """Unknown learner names should raise ValueError."""
    with pytest.raises(ValueError):
        make_main_learner(bad_name)
