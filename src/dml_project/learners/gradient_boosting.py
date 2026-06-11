"""Gradient Boosting learner wrapper for nuisance estimation."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


class GradientBoostingLearner:
    """Wrapper around :class:`sklearn.ensemble.HistGradientBoostingRegressor`."""

    def __init__(self, random_state: int | None = None) -> None:
        """Initialize a deterministic histogram gradient boosting learner.

        Args:
            random_state: Optional seed for reproducible boosting behavior.
        """
        self.random_state = random_state
        self.is_fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the gradient boosting model.

        Args:
            X: Feature matrix of shape ``(n, p)``.
            y: Target vector of shape ``(n,)``.
        """

        model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.0,
            min_samples_leaf=20,
            random_state=self.random_state,
        )
        model.fit(X, y)
        self.model_ = model
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
            raise RuntimeError("GradientBoostingLearner must be fitted before predict")
        return self.model_.predict(X)
