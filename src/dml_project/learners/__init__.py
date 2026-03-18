"""Public learner wrappers used for nuisance function estimation.

The package exposes stable wrapper classes with a common ``fit``/``predict``
interface so estimator code can swap learners without branching on sklearn
types directly.
"""

from dml_project.learners.elastic_net import ElasticNetCVLearner
from dml_project.learners.lasso import LassoCVLearner
from dml_project.learners.ols import OLSLearner

__all__ = ["OLSLearner", "LassoCVLearner", "ElasticNetCVLearner"]
