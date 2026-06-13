"""Tests for nuisance learner wrappers and factories."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.linear_model import Lasso, LassoCV

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    from dml_project import config
    from dml_project.learners.elastic_net import ElasticNetLearner
    from dml_project.learners.gradient_boosting import GradientBoostingLearner
    from dml_project.learners.lasso import LassoLearner
    from dml_project.learners.ols import OLSLearner
    from dml_project.learners.tuning import make_main_learner
except ModuleNotFoundError:
    from src.dml_project import config
    from src.dml_project.learners.elastic_net import ElasticNetLearner
    from src.dml_project.learners.gradient_boosting import GradientBoostingLearner
    from src.dml_project.learners.lasso import LassoLearner
    from src.dml_project.learners.ols import OLSLearner
    from src.dml_project.learners.tuning import make_main_learner


def _toy_data(n: int = 80, p: int = 10) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    X = rng.normal(size=(n, p))
    beta = np.array([1.0, -0.5, 0.0, 0.3, 0.0, 0.2, 0.4, 0.0, -0.2, 0.1])
    y = X @ beta + rng.normal(scale=0.1, size=n)
    return X, y


@pytest.mark.parametrize(
    "learner",
    [
        OLSLearner(),
        LassoLearner(),
        ElasticNetLearner(random_state=123),
        GradientBoostingLearner(random_state=123),
    ],
)
def test_predict_before_fit_raises_runtime_error(learner) -> None:
    """Predict before fit should raise RuntimeError."""
    X, _ = _toy_data()
    with pytest.raises(RuntimeError):
        learner.predict(X)


@pytest.mark.parametrize(
    ("learner_name", "learner_type"),
    [
        ("ols", OLSLearner),
        ("lasso", LassoLearner),
        ("elastic_net", ElasticNetLearner),
        ("gradient_boosting", GradientBoostingLearner),
    ],
)
def test_make_main_learner_returns_expected_type(
    learner_name: str,
    learner_type: type,
) -> None:
    """Factory should construct every configured learner."""

    learner = make_main_learner(learner_name, random_state=123)

    assert isinstance(learner, learner_type)


def test_config_learners_are_exact_revised_set() -> None:
    """Config should expose the revised nuisance learner set."""

    assert config.LEARNERS == [
        "ols",
        "lasso",
        "elastic_net",
        "gradient_boosting",
    ]


@pytest.mark.parametrize("learner_name", config.LEARNERS)
def test_configured_learners_fit_and_predict(learner_name: str) -> None:
    """Every configured learner should fit and predict a toy regression."""

    X, y = _toy_data()
    learner = make_main_learner(learner_name, random_state=123)

    assert hasattr(learner, "fit")
    assert hasattr(learner, "predict")

    learner.fit(X, y)
    preds = learner.predict(X[:5])

    assert preds.shape == (5,)


def test_lasso_uses_fixed_sklearn_estimator() -> None:
    """Lasso should not use internal cross-validation."""

    X, y = _toy_data()
    learner = make_main_learner("lasso", random_state=123)

    assert isinstance(learner, LassoLearner)
    assert hasattr(learner, "fit")
    assert hasattr(learner, "predict")

    learner.fit(X, y)
    predictions = learner.predict(X[:7])

    assert predictions.shape == (7,)
    assert isinstance(learner.model_, Lasso)
    assert not isinstance(learner.model_, LassoCV)


def test_lasso_uses_configured_fixed_hyperparameters() -> None:
    """Lasso should use centralized fixed hyperparameters."""

    learner = make_main_learner("lasso", random_state=456)

    assert learner.alpha == config.LASSO_ALPHA
    assert learner.max_iter == config.LASSO_MAX_ITER
    assert learner.tol == config.LASSO_TOL
    assert learner.random_state == 456

    X, y = _toy_data()
    learner.fit(X, y)

    assert learner.model_.alpha == config.LASSO_ALPHA
    assert learner.model_.random_state == 456


def test_elastic_net_uses_fixed_sklearn_estimator() -> None:
    """Elastic Net should not use internal cross-validation."""

    X, y = _toy_data()
    learner = make_main_learner("elastic_net", random_state=123)

    assert isinstance(learner, ElasticNetLearner)
    assert hasattr(learner, "fit")
    assert hasattr(learner, "predict")

    learner.fit(X, y)
    predictions = learner.predict(X[:7])

    assert predictions.shape == (7,)
    assert isinstance(learner.model_, ElasticNet)
    assert not isinstance(learner.model_, ElasticNetCV)


def test_elastic_net_uses_configured_fixed_hyperparameters() -> None:
    """Elastic Net should use centralized fixed hyperparameters."""

    learner = make_main_learner("elastic_net", random_state=456)

    assert learner.alpha == config.ELASTIC_NET_ALPHA
    assert learner.l1_ratio == config.ELASTIC_NET_L1_RATIO
    assert learner.max_iter == config.ELASTIC_NET_MAX_ITER
    assert learner.tol == config.ELASTIC_NET_TOL
    assert learner.random_state == 456

    X, y = _toy_data()
    learner.fit(X, y)

    assert learner.model_.alpha == config.ELASTIC_NET_ALPHA
    assert learner.model_.l1_ratio == config.ELASTIC_NET_L1_RATIO
    assert learner.model_.random_state == 456


def test_gradient_boosting_uses_configured_hyperparameters() -> None:
    """Gradient Boosting should use centralized runtime-conscious hyperparameters."""

    X, y = _toy_data()
    learner = make_main_learner("gradient_boosting", random_state=456)

    learner.fit(X, y)
    predictions = learner.predict(X[:7])

    assert predictions.shape == (7,)
    assert learner.model_.max_iter == config.GRADIENT_BOOSTING_MAX_ITER
    assert learner.model_.learning_rate == config.GRADIENT_BOOSTING_LEARNING_RATE
    assert learner.model_.max_leaf_nodes == config.GRADIENT_BOOSTING_MAX_LEAF_NODES
    assert (
        learner.model_.l2_regularization
        == config.GRADIENT_BOOSTING_L2_REGULARIZATION
    )
    assert learner.model_.min_samples_leaf == config.GRADIENT_BOOSTING_MIN_SAMPLES_LEAF
    assert learner.model_.random_state == 456


@pytest.mark.parametrize("learner_name", ["gradient_boosting"])
def test_stochastic_learners_are_reproducible(learner_name: str) -> None:
    """Stochastic learners should reproduce predictions for the same seed."""

    X, y = _toy_data()
    learner_1 = make_main_learner(learner_name, random_state=321)
    learner_2 = make_main_learner(learner_name, random_state=321)

    learner_1.fit(X, y)
    learner_2.fit(X, y)

    assert np.allclose(learner_1.predict(X[:5]), learner_2.predict(X[:5]))


@pytest.mark.parametrize("bad_name", ["ridge", "random_forest", ""])
def test_invalid_learner_name_raises_value_error(bad_name: str) -> None:
    """Unknown learner names should raise ValueError."""
    with pytest.raises(ValueError):
        make_main_learner(bad_name)
