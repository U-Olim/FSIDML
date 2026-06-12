"""Elastic Net learner wrapper for nuisance estimation."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import ElasticNet

from dml_project import config


class ElasticNetLearner:
    """Wrapper around fixed-parameter :class:`sklearn.linear_model.ElasticNet`."""

    def __init__(
        self,
        random_state: int | None = None,
        alpha: float = config.ELASTIC_NET_ALPHA,
        l1_ratio: float = config.ELASTIC_NET_L1_RATIO,
        max_iter: int = config.ELASTIC_NET_MAX_ITER,
        tol: float = config.ELASTIC_NET_TOL,
        cv: int | None = None,
    ) -> None:
        """Initialize a fixed-hyperparameter Elastic Net learner.

        Args:
            random_state: Optional random seed for reproducible randomness in
                sklearn's fitting routine.
            alpha: Fixed regularization strength.
            l1_ratio: Fixed L1 mixing ratio.
            max_iter: Maximum solver iterations for each fit.
            tol: Optimization tolerance.
            cv: Deprecated compatibility argument; ignored.
        """
        self.random_state = random_state
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.cv = cv
        self.is_fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the ElasticNet model.

        Args:
            X: Feature matrix of shape ``(n, p)``.
            y: Target vector of shape ``(n,)``.
        """

        model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=True,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            selection="cyclic",
        )
        model.fit(X, y)

        self.model_ = model
        self.alpha_ = float(model.alpha)
        self.l1_ratio_ = float(model.l1_ratio)
        self.is_fitted_ = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets from features using the fitted model.

        Args:
            X: Feature matrix of shape ``(n, p)``.

        Returns:
            Predicted values with shape ``(n,)``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """

        if not self.is_fitted_:
            raise RuntimeError("ElasticNetLearner must be fitted before predict")
        return self.model_.predict(X)


class ElasticNetCVLearner(ElasticNetLearner):
    """Backward-compatible alias for the fixed Elastic Net learner."""
