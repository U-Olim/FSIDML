"""Project-level constants for simulation design and reproducibility."""

from __future__ import annotations

# Run-mode replication counts.
# FAST mode is for development and quick iteration on the full design grid.
FAST_N_REP = 500
# FULL mode is for final Monte Carlo evaluation on the same grid.
FULL_N_REP = 1000

N_VALUES = [250, 500, 1000]
P_VALUES = [25, 50, 100, 150, 300]
# Outer cross-fitting folds for the simulation design.
K_VALUES = [2, 5, 10]
# Pilot scenarios use selected (n, p) cells rather than the full Cartesian grid.
PILOT_N_P_PAIRS = [
    (500, 50),
    (500, 150),
    (250, 150),
    (250, 300),
]
THETA_TRUE = 1.0

BASE_SEED = 123
# Backward-compatible default for existing estimators and learner CV settings.
DEFAULT_N_FOLDS = 2
N_FOLDS = DEFAULT_N_FOLDS
# Inner hyperparameter tuning folds are separate from outer DML cross-fitting K.
INNER_CV_FOLDS = 5

DGP_NAMES = [
    "linear_confounding",
    "quadratic_confounding",
    "interaction_confounding",
    "step_confounding",
]
LEARNERS = [
    "ols",
    "lasso",
    "elastic_net",
    "random_forest",
    "gradient_boosting",
]
