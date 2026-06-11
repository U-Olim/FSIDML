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

| Component                 | Setting                                                                 |
| ------------------------- | ----------------------------------------------------------------------- |
| Sample sizes              | n = {250, 500, 1000}                                                    |
| Covariate dimension       | p = {25, 50, 100, 150, 300}                                             |
| Cross-fitting folds       | K = {2, 5, 10}                                                          |
| Nuisance estimators       | OLS, Lasso, Elastic Net, Random Forest, Gradient Boosting               |
| Data generating processes | linear_confounding, quadratic_confounding, interaction_confounding, step_confounding |
| Replications              | `1000` per full-simulation scenario                                     |

The DGPs are adapted benchmark designs with linear, quadratic/U-shaped, interaction, and step-function confounding. The contribution is not inventing new theoretical DGPs; it is studying how these nuisance structures interact with fold-level dimensionality and learner stability in finite-sample DML inference.

Total number of full scenarios:

4 (DGP) x 3 (n) x 5 (p) x 3 (K) x 5 (learners) = 900

Full simulation mode requires many model fits and can take substantial time depending on hardware. Smoke and pilot workflow modes are available for validation without running the full Monte Carlo design.

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
