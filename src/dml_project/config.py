"""Project-level constants for simulation design and reproducibility."""

from __future__ import annotations

# Run-mode replication counts.
# FAST mode is for development and quick iteration on the full design grid.
FAST_N_REP = 500
# FULL mode is for final Monte Carlo evaluation on the same grid.
FULL_N_REP = 1000

N_VALUES = [200, 300, 400]
P_VALUES = [100, 150]
THETA_TRUE = 1.0

BASE_SEED = 123
N_FOLDS = 2

DGP_NAMES = ["linear_baseline", "linear_sparse_correlated"]
LEARNERS = ["ols", "lasso", "elastic_net"]
