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
    from dml_project.learners.gradient_boosting import GradientBoostingLearner
    from dml_project.learners.lasso import LassoCVLearner
    from dml_project.learners.ols import OLSLearner
    from dml_project.learners.random_forest import RandomForestLearner
    from dml_project.learners.tuning import make_main_learner
except ModuleNotFoundError:
    from src.dml_project import config
    from src.dml_project.learners.elastic_net import ElasticNetCVLearner
    from src.dml_project.learners.gradient_boosting import GradientBoostingLearner
    from src.dml_project.learners.lasso import LassoCVLearner
    from src.dml_project.learners.ols import OLSLearner
    from src.dml_project.learners.random_forest import RandomForestLearner
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
        LassoCVLearner(),
        ElasticNetCVLearner(random_state=123),
        RandomForestLearner(random_state=123),
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
        ("lasso", LassoCVLearner),
        ("elastic_net", ElasticNetCVLearner),
        ("random_forest", RandomForestLearner),
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
        "random_forest",
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


def test_factory_uses_inner_cv_for_penalized_learners() -> None:
    """Penalized learners should use inner tuning CV, not outer DML folds."""

    assert config.INNER_CV_FOLDS == 5
    lasso = make_main_learner("lasso")
    elastic = make_main_learner("elastic_net")
    assert isinstance(lasso, LassoCVLearner)
    assert lasso.cv == config.INNER_CV_FOLDS
    assert lasso.cv != config.N_FOLDS
    assert isinstance(elastic, ElasticNetCVLearner)
    assert elastic.cv == config.INNER_CV_FOLDS
    assert elastic.cv != config.N_FOLDS


@pytest.mark.parametrize("learner_name", ["random_forest", "gradient_boosting"])
def test_stochastic_learners_are_reproducible(learner_name: str) -> None:
    """Stochastic learners should reproduce predictions for the same seed."""

    X, y = _toy_data()
    learner_1 = make_main_learner(learner_name, random_state=321)
    learner_2 = make_main_learner(learner_name, random_state=321)

    learner_1.fit(X, y)
    learner_2.fit(X, y)

    assert np.allclose(learner_1.predict(X[:5]), learner_2.predict(X[:5]))


@pytest.mark.parametrize("bad_name", ["ridge", ""])
def test_invalid_learner_name_raises_value_error(bad_name: str) -> None:
    """Unknown learner names should raise ValueError."""
    with pytest.raises(ValueError):
        make_main_learner(bad_name)
