"""Double Machine Learning estimator for partially linear regression."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from sklearn.model_selection import KFold

from dml_project import config


class DMLPLR:
    """DML estimator for the PLR model with cross-fitting.

    Partially linear model:
    Y = theta_0 D + g_0(X) + epsilon
    D = m_0(X) + v
    with E[epsilon | X, D] = 0 and E[v | X] = 0.

    Estimation:
    - Fit g_hat(X) and m_hat(X) by K-fold cross-fitting.
    - Form residuals:
      Y_res = Y - g_hat(X)
      D_res = D - m_hat(X)
    - Solve the orthogonal score moment
      E[D_res * (Y_res - theta * D_res)] = 0
      giving
      theta_hat = mean(D_res * Y_res) / mean(D_res^2)
    """

    def __init__(
        self,
        learner_g,
        learner_m,
        n_folds: int = config.N_FOLDS,
        random_state: int | None = None,
    ) -> None:
        """Initialize a DML-PLR estimator.

        Args:
            learner_g: Regressor used for the outcome nuisance function
                ``g_0(X) = E[Y | X]``. Must implement ``fit`` and ``predict``.
            learner_m: Regressor used for the treatment nuisance function
                ``m_0(X) = E[D | X]``. Must implement ``fit`` and ``predict``.
            n_folds: Number of folds used for cross-fitting.
            random_state: Optional random seed for shuffled K-fold splits.
        """
        self.learner_g = learner_g
        self.learner_m = learner_m
        self.n_folds = n_folds
        self.random_state = random_state
        self.is_fitted_ = False

    def fit(self, Y: np.ndarray, D: np.ndarray, X: np.ndarray) -> None:
        """Fit the DML-PLR estimator and compute inference quantities.

        Args:
            Y: Outcome vector with shape ``(n,)``.
            D: Treatment vector with shape ``(n,)``.
            X: Covariate matrix with shape ``(n, p)``.

        Returns:
            ``None``. Fitted results are stored on the instance via ``*_``
            attributes, including ``theta_hat_``, ``se_``, and confidence
            interval endpoints.

        Raises:
            ValueError: If inputs have invalid shapes, sample sizes mismatch,
                or residualized treatment variance is near zero.
            RuntimeError: If cross-fitting integrity checks fail.
        """

        Y = np.asarray(Y)
        D = np.asarray(D)
        X = np.asarray(X)

        if Y.ndim != 1:
            raise ValueError("Y must be a 1D array")
        if D.ndim != 1:
            raise ValueError("D must be a 1D array")
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[0] != Y.shape[0] or D.shape[0] != Y.shape[0]:
            raise ValueError("Y, D, and X must have matching sample size")
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2")

        kf = KFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        g_hat_all = np.empty_like(Y, dtype=float)
        m_hat_all = np.empty_like(D, dtype=float)
        oof_counts = np.zeros(Y.shape[0], dtype=int)
        fold_indices: list[tuple[np.ndarray, np.ndarray]] = []
        fold_diagnostics: list[dict[str, int]] = []
        learner_g_fold_names: list[str] = []
        learner_m_fold_names: list[str] = []

        for fold_id, (train_idx, valid_idx) in enumerate(kf.split(X)):
            fold_indices.append((train_idx.copy(), valid_idx.copy()))
            learner_g_fold = deepcopy(self.learner_g)
            learner_m_fold = deepcopy(self.learner_m)
            learner_g_fold_names.append(type(learner_g_fold).__name__)
            learner_m_fold_names.append(type(learner_m_fold).__name__)

            overlap_count = int(np.intersect1d(train_idx, valid_idx).size)
            if overlap_count != 0:
                raise RuntimeError(
                    f"Cross-fitting leakage detected in fold {fold_id}: overlap_count={overlap_count}"
                )
            fold_diagnostics.append(
                {
                    "fold_id": int(fold_id),
                    "train_size": int(train_idx.size),
                    "test_size": int(valid_idx.size),
                    "overlap_count": overlap_count,
                }
            )

            learner_g_fold.fit(X[train_idx], Y[train_idx])
            learner_m_fold.fit(X[train_idx], D[train_idx])

            g_hat_all[valid_idx] = learner_g_fold.predict(X[valid_idx])
            m_hat_all[valid_idx] = learner_m_fold.predict(X[valid_idx])
            oof_counts[valid_idx] += 1

        if not np.all(oof_counts == 1):
            raise RuntimeError("Cross-fitting failed: each sample must be predicted out-of-fold once")

        Y_res = Y - g_hat_all
        D_res = D - m_hat_all

        denominator = np.mean(D_res**2)
        if np.isclose(denominator, 0.0):
            raise ValueError("Mean squared treatment residual is too close to zero")

        theta_hat = np.mean(D_res * Y_res) / denominator
        psi = D_res * (Y_res - theta_hat * D_res)
        sigma2_hat = np.mean(psi**2) / (denominator**2)
        se = float(np.sqrt(sigma2_hat / Y.shape[0]))
        ci_lower = float(theta_hat - 1.96 * se)
        ci_upper = float(theta_hat + 1.96 * se)

        self.theta_hat_ = float(theta_hat)
        self.se_ = se
        self.ci_lower_ = ci_lower
        self.ci_upper_ = ci_upper
        self.g_hat_ = g_hat_all
        self.m_hat_ = m_hat_all
        self.oof_counts_ = oof_counts
        self.fold_indices_ = fold_indices
        self.fold_diagnostics_ = fold_diagnostics
        self.learner_g_name_ = type(self.learner_g).__name__
        self.learner_m_name_ = type(self.learner_m).__name__
        self.learner_g_fold_names_ = learner_g_fold_names
        self.learner_m_fold_names_ = learner_m_fold_names
        self.is_fitted_ = True

    def predict_effect(self) -> float:
        """Return the fitted treatment-effect estimate ``theta_hat_``.

        Returns:
            Estimated causal effect.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """

        if not self.is_fitted_:
            raise RuntimeError("DMLPLR must be fitted before predict_effect")
        return self.theta_hat_
