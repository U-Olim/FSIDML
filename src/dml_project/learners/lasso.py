"""Lasso-based learner wrapper for nuisance estimation."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LassoCV

from dml_project import config


class LassoCVLearner:
    """Wrapper around :class:`sklearn.linear_model.LassoCV`.

    After fitting, the selected regularization strength is available via
    ``alpha_``.
    """

    def __init__(
        self,
        cv: int = config.INNER_CV_FOLDS,
        random_state: int | None = None,
        max_iter: int = 10_000,
    ) -> None:
        """Initialize a cross-validated Lasso learner.

        Args:
            cv: Number of folds used internally by ``LassoCV``.
            random_state: Optional random seed for reproducible coordinate
                descent randomness and CV shuffling behavior inside sklearn.
            max_iter: Maximum solver iterations for each fit.
        """
        self.cv = cv
        self.random_state = random_state
        self.max_iter = max_iter
        self.is_fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the LassoCV model.

        Args:
            X: Feature matrix of shape ``(n, p)``.
            y: Target vector of shape ``(n,)``.
        """

        model = LassoCV(
            cv=self.cv,
            random_state=self.random_state,
            max_iter=self.max_iter,
            selection="random",
            n_jobs=1,
        )
        model.fit(X, y)

        self.model_ = model
        self.alpha_ = float(model.alpha_)
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
            raise RuntimeError("LassoCVLearner must be fitted before predict")
        return self.model_.predict(X)
