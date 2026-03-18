"""Ordinary least squares learner wrapper for nuisance estimation."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression


class OLSLearner:
    """Thin wrapper around :class:`sklearn.linear_model.LinearRegression`.

    The wrapper exposes a consistent ``fit``/``predict`` interface that matches
    other learner classes in this project.
    """

    def __init__(self) -> None:
        self.is_fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the OLS model.

        Args:
            X: Feature matrix of shape ``(n, p)``.
            y: Target vector of shape ``(n,)``.
        """

        model = LinearRegression()
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
            raise RuntimeError("OLSLearner must be fitted before predict")
        return self.model_.predict(X)
