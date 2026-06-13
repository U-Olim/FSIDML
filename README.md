# Finite-Sample Inference in Double Machine Learning

This repository contains a simulation-only study of finite-sample inference in
Double Machine Learning (DML) under cross-fitting. The project focuses on how
fold-level dimensionality affects nuisance estimation, treatment-effect
accuracy, and confidence-interval coverage.

## Project Structure

- `src/dml_project/`: DGPs, nuisance learners, estimator, and simulation pipeline.
- `tasks/`: `pytask` workflow definitions for simulations, aggregation, tables, and figures.
- `tests/`: Unit and integration tests for the simulation workflow.
- `documents/paper/`: Paper source files.
- `documents/outputs/`: Generated outputs from smoke or full runs.

## Final Simulation Design

| Component | Setting |
| --- | --- |
| Sample sizes | n = {250, 500, 1000} |
| Covariate dimensions | p = {25, 50, 100, 150, 300} |
| Cross-fitting folds | K = {2, 5, 10} |
| DGPs | dense_linear_independent, sparse_linear_independent, sparse_linear_correlated, weak_signal_sparse |
| Nuisance learners | OLS, Lasso, Elastic Net, Gradient Boosting |
| Smoke replications | 10 per scenario |
| Full replications | 1000 per scenario |

The active grid contains `3 * 5 * 3 * 4 * 4 = 720` scenarios. Smoke mode
therefore produces 7,200 simulation rows, and full mode produces 720,000 rows.

## Workflow Modes

- `smoke`: full scenario grid with 10 replications for quick checks.
- `full`: full scenario grid with 1000 replications for final results.

## Run Commands

Install dependencies:

```bash
pixi install
```

Run tests:

```bash
pixi run pytest
```

Collect tasks without running them:

```bash
$env:DML_RUN_MODE = "smoke"
pixi run python -m pytask collect
```

Run the current workflow:

```bash
pixi run pytask
```
