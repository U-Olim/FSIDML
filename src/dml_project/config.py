"""Project-level constants for simulation design and reproducibility."""

from __future__ import annotations

from pathlib import Path

# Run-mode replication counts.
# Smoke mode checks the full scenario grid with a low replication count.
SMOKE_N_REPLICATIONS = 10
# Full mode is for final Monte Carlo evaluation on the same grid.
N_REPLICATIONS = 1000
FULL_N_REPLICATIONS = N_REPLICATIONS
# Backward-compatible alias for existing scenario defaults.
FULL_N_REP = N_REPLICATIONS

N_VALUES = [250, 500, 1000]
P_VALUES = [25, 50, 100, 150, 300]
# Outer cross-fitting folds for the simulation design.
K_VALUES = [2, 5, 10]
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "documents" / "outputs"
RAW_RESULTS_DIR = OUTPUT_DIR / "raw"
AGGREGATED_RESULTS_DIR = OUTPUT_DIR / "aggregated"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
