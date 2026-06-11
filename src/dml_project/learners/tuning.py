"""Factory helpers for constructing configured nuisance learners."""

from __future__ import annotations

from dml_project import config
from dml_project.learners.elastic_net import ElasticNetCVLearner
from dml_project.learners.gradient_boosting import GradientBoostingLearner
from dml_project.learners.lasso import LassoCVLearner
from dml_project.learners.ols import OLSLearner
from dml_project.learners.random_forest import RandomForestLearner


def make_main_learner(name: str, random_state: int | None = None):
    """Create a configured learner instance by canonical name.

    Args:
        name: Learner identifier from ``config.LEARNERS``.
        random_state: Optional seed passed to randomized learners.

    Returns:
        An initialized learner object implementing ``fit`` and ``predict``.

    Raises:
        ValueError: If ``name`` is not a supported learner key.
    """

    if name == "ols":
        return OLSLearner()
    if name == "lasso":
        return LassoCVLearner(
            cv=config.INNER_CV_FOLDS,
            random_state=random_state,
        )
    if name == "elastic_net":
        return ElasticNetCVLearner(
            cv=config.INNER_CV_FOLDS,
            random_state=random_state,
        )
    if name == "random_forest":
        return RandomForestLearner(random_state=random_state)
    if name == "gradient_boosting":
        return GradientBoostingLearner(random_state=random_state)
    raise ValueError(f"Unknown learner name: {name}")
