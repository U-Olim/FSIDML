"""Random Forest learner wrapper for nuisance estimation."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor


class RandomForestLearner:
    """Wrapper around :class:`sklearn.ensemble.RandomForestRegressor`."""

    def __init__(self, random_state: int | None = None) -> None:
        """Initialize a deterministic Random Forest learner.

        Args:
            random_state: Optional seed for reproducible tree construction.
        """
        self.random_state = random_state
        self.is_fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the Random Forest model.

        Args:
            X: Feature matrix of shape ``(n, p)``.
            y: Target vector of shape ``(n,)``.
        """

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=self.random_state,
            n_jobs=-1,
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
            raise RuntimeError("RandomForestLearner must be fitted before predict")
        return self.model_.predict(X)
