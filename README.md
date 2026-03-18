# Finite-Sample Inference in Double Machine Learning: OLS Instability under Cross-Fitting

This repository contains code to reproduce the simulation and empirical results in "Finite-Sample Inference in Double Machine Learning: OLS Instability under Cross-Fitting". It implements Double Machine Learning estimators with alternative nuisance learners and evaluates finite-sample inference performance

All results in the paper can be reproduced using the scripts provided in this repository. The code generates simulation results and empirical estimates from raw data.

## Repository Structure

- `src/dml_project/`: Core implementation (DGPs, learners, estimators, simulation pipeline, utilities).
- `tasks/`: `pytask` workflow definitions for simulations, tables, figures, and empirical outputs.
- `tests/`: Unit and integration tests for core modules and task logic.
- `documents/paper/`: Paper source and rendered outputs.
- `documents/real_data_401k/`: 401(k) empirical dataset used in the application.
- `pixi.toml`, `pyproject.toml`: Environment, dependencies, and project configuration.

## Simulation Design

We conduct a Monte Carlo simulation where many artificial datasets are generated and the DML estimator is applied repeatedly.

| Component                 | Setting                                   |
| ------------------------- | ----------------------------------------- |
| Sample sizes              | n = {200, 300, 400}                       |
| Covariate dimension       | p = {100, 150}                            |
| Nuisance estimators       | OLS, Lasso, Elastic Net                   |
| Data generating processes | linear_baseline, linear_sparse_correlated |
| Replications              | `1000` per scenario                       |

Total number of scenarios:

2 (DGP) x 3 (n) x 2 (p) x 3 (learners) = 36

Each simulation scenario is repeated 1000 times. R = 1000 requires many thousands of model fits and can take hours depending on hardware.

Total Monte Carlo estimations in FULL mode: 36 x 1000 = 36 000

This is the total number of simulated estimation runs across all scenarios and replications.

## Real Data Application

In addition to the simulation study, the project includes an empirical application using real data:

- Dataset: 1991 Survey of Income and Program Participation (SIPP) 401(k) data
- File used in this repository: `documents/real_data_401k/sipp_1991.csv`

This real-data exercise complements the Monte Carlo results by showing how the estimators behave on observed household financial data.

## How the Simulation Works

1. Generate artificial data from a known DGP.
2. Estimate the treatment effect using DML.
3. Compare estimated effects with the true parameter.
4. Repeat many times and summarize performance metrics.

If an estimator is reliable:

- Estimates should be close to the true effect.
- Confidence intervals should contain the true effect around 95% of the time.

## How to Run the Project

Install dependencies:

```bash
pixi install
```

Run tests:

```bash
pixi run pytest
```

Run full simulation pipeline:

```bash
pixi run pytask
```

## Rendering the Paper

```bash
pixi run paper-pdf
```
