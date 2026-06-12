"""Lasso-based learner wrapper for nuisance estimation."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Lasso

from dml_project import config


class LassoLearner:
    """Wrapper around fixed-parameter :class:`sklearn.linear_model.Lasso`."""

    def __init__(
        self,
        random_state: int | None = None,
        alpha: float = config.LASSO_ALPHA,
        max_iter: int = config.LASSO_MAX_ITER,
        tol: float = config.LASSO_TOL,
        cv: int | None = None,
    ) -> None:
        """Initialize a fixed-hyperparameter Lasso learner.

        Args:
            random_state: Optional random seed for reproducible coordinate
                descent randomness inside sklearn.
            alpha: Fixed regularization strength.
            max_iter: Maximum solver iterations for each fit.
            tol: Optimization tolerance.
            cv: Deprecated compatibility argument; ignored.
        """
        self.random_state = random_state
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.cv = cv
        self.is_fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the Lasso model.

        Args:
            X: Feature matrix of shape ``(n, p)``.
            y: Target vector of shape ``(n,)``.
        """

        model = Lasso(
            alpha=self.alpha,
            fit_intercept=True,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            selection="cyclic",
        )
        model.fit(X, y)

        self.model_ = model
        self.alpha_ = float(model.alpha)
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
            raise RuntimeError("LassoLearner must be fitted before predict")
        return self.model_.predict(X)


class LassoCVLearner(LassoLearner):
    """Backward-compatible alias for the fixed Lasso learner."""
