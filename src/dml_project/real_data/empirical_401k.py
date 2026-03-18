"""Empirical application on the 1991 SIPP 401(k) dataset."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from dml_project import config
from dml_project.estimators.dml_plr import DMLPLR


_BASE_X_COLUMNS = ["age", "inc", "fsize", "educ", "pira", "hown", "marr", "db", "twoearn"]


@dataclass(frozen=True)
class OLSResult:
    """Container for OLS treatment-effect inference."""

    theta_hat: float
    se: float
    ci_lower: float
    ci_upper: float


def prepare_401k_data(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare outcome, treatment, and high-dimensional controls for the empirical run.

    The specification follows the project requirement and drops observations with
    nonpositive income because log(inc) is part of the control set.
    """

    required = {"net_tfa", "e401", *_BASE_X_COLUMNS}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"401(k) data is missing required columns: {missing}")

    work = data.copy()
    work = work.loc[work["inc"] > 0].reset_index(drop=True)
    if work.empty:
        raise ValueError("No observations left after filtering to inc > 0 for log(inc).")

    work["age_sq"] = work["age"] ** 2
    work["inc_sq"] = work["inc"] ** 2
    work["educ_sq"] = work["educ"] ** 2
    work["log_inc"] = np.log(work["inc"])
    work["age_inc"] = work["age"] * work["inc"]
    work["educ_inc"] = work["educ"] * work["inc"]
    work["marr_inc"] = work["marr"] * work["inc"]

    x_columns = [
        *_BASE_X_COLUMNS,
        "age_sq",
        "inc_sq",
        "educ_sq",
        "log_inc",
        "age_inc",
        "educ_inc",
        "marr_inc",
    ]
    y = work["net_tfa"].to_numpy(dtype=float)
    d = work["e401"].to_numpy(dtype=float)
    x = work[x_columns].to_numpy(dtype=float)
    return y, d, x


def _fit_ols_with_hc1_se(y: np.ndarray, d: np.ndarray, x: np.ndarray) -> OLSResult:
    """Estimate OLS effect of D on Y with controls X and HC1 robust SE."""

    n = y.shape[0]
    x_scaled = StandardScaler().fit_transform(x)
    design = np.column_stack([np.ones(n), d, x_scaled])
    gram = design.T @ design
    gram_inv = np.linalg.pinv(gram)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta

    k = design.shape[1]
    scale = n / max(n - k, 1)
    meat = design.T @ ((resid**2)[:, None] * design)
    vcov = scale * gram_inv @ meat @ gram_inv
    se = float(np.sqrt(max(vcov[1, 1], 0.0)))
    theta = float(beta[1])
    ci_lower = float(theta - 1.96 * se)
    ci_upper = float(theta + 1.96 * se)
    return OLSResult(theta_hat=theta, se=se, ci_lower=ci_lower, ci_upper=ci_upper)


def _fit_dml(y: np.ndarray, d: np.ndarray, x: np.ndarray, learner, seed: int) -> tuple[float, float, float, float]:
    """Fit DML-PLR with a provided nuisance learner object."""

    dml = DMLPLR(
        learner_g=learner,
        learner_m=learner,
        n_folds=config.N_FOLDS,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        dml.fit(Y=y, D=d, X=x)
    return dml.theta_hat_, dml.se_, dml.ci_lower_, dml.ci_upper_


def run_401k_empirical_estimators(data: pd.DataFrame, seed: int = config.BASE_SEED) -> pd.DataFrame:
    """Run OLS, DML-LASSO, and DML-Elastic Net on prepared 401(k) data."""

    y, d, x = prepare_401k_data(data)
    ols = _fit_ols_with_hc1_se(y=y, d=d, x=x)

    lasso_learner = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                LassoCV(
                    cv=config.N_FOLDS,
                    random_state=seed,
                    max_iter=10_000,
                    selection="random",
                    n_jobs=1,
                ),
            ),
        ]
    )
    lasso_theta, lasso_se, lasso_low, lasso_high = _fit_dml(y=y, d=d, x=x, learner=lasso_learner, seed=seed)

    en_learner = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                ElasticNetCV(
                    cv=config.N_FOLDS,
                    random_state=seed,
                    max_iter=10_000,
                    selection="random",
                    n_jobs=1,
                ),
            ),
        ]
    )
    en_theta, en_se, en_low, en_high = _fit_dml(y=y, d=d, x=x, learner=en_learner, seed=seed)

    return pd.DataFrame(
        [
            {
                "estimator": "OLS",
                "theta_hat": ols.theta_hat,
                "se": ols.se,
                "ci_lower": ols.ci_lower,
                "ci_upper": ols.ci_upper,
            },
            {
                "estimator": "DML-LASSO",
                "theta_hat": float(lasso_theta),
                "se": float(lasso_se),
                "ci_lower": float(lasso_low),
                "ci_upper": float(lasso_high),
            },
            {
                "estimator": "DML-Elastic Net",
                "theta_hat": float(en_theta),
                "se": float(en_se),
                "ci_lower": float(en_low),
                "ci_upper": float(en_high),
            },
        ]
    )


def create_coef_plot(results: pd.DataFrame, output_png: str, output_pdf: str) -> None:
    """Create Figure 14 coefficient plot with estimator-specific colors."""

    required = {"estimator", "theta_hat", "ci_lower", "ci_upper"}
    missing = sorted(required.difference(results.columns))
    if missing:
        raise ValueError(f"Results for coefficient plot are missing columns: {missing}")

    order = ["OLS", "DML-LASSO", "DML-Elastic Net"]
    color_map = {"OLS": "red", "DML-LASSO": "blue", "DML-Elastic Net": "green"}
    plot_df = results.set_index("estimator").reindex(order).reset_index()
    if plot_df["theta_hat"].isna().any():
        raise ValueError("Coefficient plot requires all estimators: OLS, DML-LASSO, DML-Elastic Net.")

    y_pos = np.arange(len(plot_df))
    center = plot_df["theta_hat"].to_numpy(dtype=float)
    lower = plot_df["ci_lower"].to_numpy(dtype=float)
    upper = plot_df["ci_upper"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for idx, estimator in enumerate(plot_df["estimator"]):
        ax.errorbar(
            center[idx],
            y_pos[idx],
            xerr=np.array([[center[idx] - lower[idx]], [upper[idx] - center[idx]]]),
            fmt="o",
            color=color_map[estimator],
            ecolor=color_map[estimator],
            elinewidth=1.3,
            capsize=3.0,
            markersize=5.5,
        )

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_yticks(y_pos, plot_df["estimator"])
    ax.set_xlabel("Treatment Effect on net_tfa")
    ax.set_ylabel("Estimator")
    ax.set_title("Figure 14: 401(k) Treatment Effect Estimates")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    fig.savefig(output_pdf, dpi=300)
    plt.close(fig)
