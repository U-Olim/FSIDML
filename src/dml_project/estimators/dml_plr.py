"""Double Machine Learning estimator for partially linear regression."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from sklearn.model_selection import KFold

from dml_project import config

_IDENTIFICATION_TOL = 1e-12
_IDENTIFICATION_ERROR = (
    "Near-zero residualized treatment variance; DML PLR estimate is not identified."
)


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R2 and return NaN when the held-out target is constant."""

    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denominator == 0.0:
        return float("nan")
    numerator = float(np.sum((y_true - y_pred) ** 2))
    return float(1.0 - numerator / denominator)


def _design_diagnostics(X_train: np.ndarray) -> dict[str, float | bool]:
    """Compute fold-level diagnostics for the nuisance training design."""

    fold_train_size = X_train.shape[0]
    gram = X_train.T @ X_train / fold_train_size
    eigenvalues = np.linalg.eigvalsh(gram)
    return {
        "fold_ratio": float(X_train.shape[1] / fold_train_size),
        "condition_number": float(np.linalg.cond(gram)),
        "min_eigenvalue": float(eigenvalues[0]),
        "rank_deficient": bool(np.linalg.matrix_rank(X_train) < min(X_train.shape)),
    }


def _aggregate_fold_diagnostics(
    fold_diagnostics: list[dict[str, float | int | bool]],
) -> dict[str, float]:
    """Aggregate fold diagnostics with NaN-aware means where relevant."""

    def values(key: str) -> np.ndarray:
        return np.asarray([fold[key] for fold in fold_diagnostics], dtype=float)

    fold_train_sizes = values("fold_train_size")
    fold_test_sizes = values("fold_test_size")
    fold_ratios = values("fold_ratio")
    condition_numbers = values("condition_number")
    min_eigenvalues = values("min_eigenvalue")

    return {
        "mean_fold_train_size": float(np.nanmean(fold_train_sizes)),
        "mean_fold_test_size": float(np.nanmean(fold_test_sizes)),
        "mean_fold_ratio": float(np.nanmean(fold_ratios)),
        "max_fold_ratio": float(np.nanmax(fold_ratios)),
        "mean_condition_number": float(np.nanmean(condition_numbers)),
        "max_condition_number": float(np.nanmax(condition_numbers)),
        "mean_min_eigenvalue": float(np.nanmean(min_eigenvalues)),
        "min_min_eigenvalue": float(np.nanmin(min_eigenvalues)),
        "rank_deficiency_rate": float(np.nanmean(values("rank_deficient"))),
        "mean_nuisance_mse_y": float(np.nanmean(values("nuisance_mse_y"))),
        "mean_nuisance_mse_d": float(np.nanmean(values("nuisance_mse_d"))),
        "mean_nuisance_r2_y": float(np.nanmean(values("nuisance_r2_y"))),
        "mean_nuisance_r2_d": float(np.nanmean(values("nuisance_r2_d"))),
    }


def _set_random_state(learner, random_state: int | None) -> None:
    """Set a learner seed when the wrapper exposes a random_state attribute."""

    if hasattr(learner, "random_state"):
        learner.random_state = random_state


class DMLPLR:
    """DML estimator for the PLR model with cross-fitting.

    Partially linear model:
    Y = theta_0 D + g_0(X) + U
    D = r_0(X) + V
    with E[U | X, D] = 0 and E[V | X] = 0.

    Estimation:
    - Fit ell_hat(X) = E[Y | X] and r_hat(X) = E[D | X] by cross-fitting.
    - Form residuals:
      y_residual = Y - ell_hat(X)
      d_residual = D - r_hat(X)
    - Solve the orthogonal score moment
      E[d_residual * (y_residual - theta * d_residual)] = 0
      giving
      theta_hat = sum(d_residual * y_residual) / sum(d_residual^2)
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
            learner_g: Regressor used for ``ell_0(X) = E[Y | X]``. Must
                implement ``fit`` and ``predict``.
            learner_m: Regressor used for ``r_0(X) = E[D | X]``. Must
                implement ``fit`` and ``predict``.
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
            RuntimeError: If cross-fitting integrity validation fails.
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

        ell_hat = np.empty_like(Y, dtype=float)
        r_hat = np.empty_like(D, dtype=float)
        oof_counts = np.zeros(Y.shape[0], dtype=int)
        fold_indices: list[tuple[np.ndarray, np.ndarray]] = []
        fold_diagnostics: list[dict[str, float | int | bool]] = []
        learner_g_fold_names: list[str] = []
        learner_m_fold_names: list[str] = []

        for fold_id, (train_idx, valid_idx) in enumerate(kf.split(X)):
            fold_indices.append((train_idx.copy(), valid_idx.copy()))
            learner_g_fold = deepcopy(self.learner_g)
            learner_m_fold = deepcopy(self.learner_m)
            if self.random_state is not None:
                _set_random_state(learner_g_fold, self.random_state + 1000 + fold_id)
                _set_random_state(learner_m_fold, self.random_state + 2000 + fold_id)
            learner_g_fold_names.append(type(learner_g_fold).__name__)
            learner_m_fold_names.append(type(learner_m_fold).__name__)

            overlap_count = int(np.intersect1d(train_idx, valid_idx).size)
            if overlap_count != 0:
                raise RuntimeError(
                    f"Cross-fitting leakage detected in fold {fold_id}: overlap_count={overlap_count}"
                )

            learner_g_fold.fit(X[train_idx], Y[train_idx])
            learner_m_fold.fit(X[train_idx], D[train_idx])

            ell_hat[valid_idx] = learner_g_fold.predict(X[valid_idx])
            r_hat[valid_idx] = learner_m_fold.predict(X[valid_idx])
            y_test = Y[valid_idx]
            d_test = D[valid_idx]
            ell_hat_test = ell_hat[valid_idx]
            r_hat_test = r_hat[valid_idx]
            fold_diagnostic = {
                "fold_id": int(fold_id),
                "fold_train_size": int(train_idx.size),
                "fold_test_size": int(valid_idx.size),
                "overlap_count": overlap_count,
                "nuisance_mse_y": float(np.mean((y_test - ell_hat_test) ** 2)),
                "nuisance_mse_d": float(np.mean((d_test - r_hat_test) ** 2)),
                "nuisance_r2_y": _safe_r2(y_test, ell_hat_test),
                "nuisance_r2_d": _safe_r2(d_test, r_hat_test),
            }
            fold_diagnostic.update(_design_diagnostics(X[train_idx]))
            fold_diagnostics.append(fold_diagnostic)
            oof_counts[valid_idx] += 1

        if not np.all(oof_counts == 1):
            raise RuntimeError("Cross-fitting failed: each sample must be predicted out-of-fold once")

        y_residual = Y - ell_hat
        d_residual = D - r_hat

        denominator = float(np.sum(d_residual**2))
        if denominator <= _IDENTIFICATION_TOL:
            raise ValueError(_IDENTIFICATION_ERROR)

        theta_hat = float(np.sum(d_residual * y_residual) / denominator)
        structural_residual = y_residual - theta_hat * d_residual
        psi = structural_residual * d_residual
        j_hat = float(np.mean(d_residual**2))
        if j_hat <= _IDENTIFICATION_TOL:
            raise ValueError(_IDENTIFICATION_ERROR)
        var_hat = float(np.mean(psi**2) / (j_hat**2))
        se = float(np.sqrt(var_hat / Y.shape[0]))
        ci_lower = float(theta_hat - 1.96 * se)
        ci_upper = float(theta_hat + 1.96 * se)

        self.theta_hat_ = theta_hat
        self.se_ = se
        self.ci_lower_ = ci_lower
        self.ci_upper_ = ci_upper
        self.ell_hat_ = ell_hat
        self.r_hat_ = r_hat
        self.y_residual_ = y_residual
        self.d_residual_ = d_residual
        self.g_hat_ = ell_hat
        self.m_hat_ = r_hat
        self.oof_counts_ = oof_counts
        self.fold_indices_ = fold_indices
        self.fold_diagnostics_ = fold_diagnostics
        self.aggregate_diagnostics_ = _aggregate_fold_diagnostics(fold_diagnostics)
        self.learner_g_name_ = type(self.learner_g).__name__
        self.learner_m_name_ = type(self.learner_m).__name__
        self.learner_g_fold_names_ = learner_g_fold_names
        self.learner_m_fold_names_ = learner_m_fold_names
        self.theta_hat = self.theta_hat_
        self.se = self.se_
        self.ci_lower = self.ci_lower_
        self.ci_upper = self.ci_upper_
        self.ell_hat = self.ell_hat_
        self.r_hat = self.r_hat_
        self.y_residual = self.y_residual_
        self.d_residual = self.d_residual_
        self.g_hat = self.g_hat_
        self.m_hat = self.m_hat_
        self.fold_diagnostics = self.fold_diagnostics_
        self.aggregate_diagnostics = self.aggregate_diagnostics_
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
