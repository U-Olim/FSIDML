# RMSE Check Report

## 1) RMSE computation trace
- RMSE is computed in `src/dml_project/simulation/metrics.py` via `sqrt(mean((theta_hat - theta_true)^2))`.
- Scenario aggregation happens in `src/dml_project/simulation/aggregate.py` (`aggregate_results`) and is merged into `scenario_summary.csv` by `tasks/task_aggregate.py`.
- `theta_true` is fixed at `1.0` through `src/dml_project/config.py` (`THETA_TRUE = 1.0`) and enforced by `Scenario.__post_init__` in `src/dml_project/simulation/scenario.py`.

## 2) Reproducibility of suspicious OLS RMSE cells
- Recomputed RMSE from raw replications matches aggregated values (max absolute difference across checked cells: 0.000000000000).
- The large OLS values at `(n=200,p=100)` and `(n=300,p=150)` are reproducible in both DGPs.

## 3) Data-quality and aggregation-bug checks
- Duplicate `(scenario_id, replication)` pairs: 0.
- Missing core (`theta_hat`, `theta_true`, `se`) rows: 0.
- Unique replication counts per scenario: [1000].
- Scenario-name mismatch count versus design keys: 0.
- No evidence of wrong merge/join or scenario-label mismatch in checked outputs.
- No evidence of accidental use of variance or SE in place of RMSE (formula-matched recomputation).

## 4) Seed logic and replication count across learners
- Replication count is consistent across learners (`1000` for every scenario).
- Seed-generation logic is consistent across learners (same deterministic function), but common random numbers are **not** used across learners because `scenario_id` enters seed generation.
- Common data seed across learners for same `(dgp,n,p,replication)`: False.

## 5) Likely source of large OLS RMSE
- The high-RMSE cells are exactly where fold-level OLS nuisance fits are weakest under cross-fitting: `n=200,p=100` and `n=300,p=150` imply training-fold sizes `100` and `150`, i.e., `n_train ~= p`.
- In these cells, OLS shows heavy-tailed `theta_hat` with many extreme absolute errors (counts across four OLS checked cells: `|error|>2`: 88, `|error|>5`: 5, `|error|>10`: 0).
- Calibration diagnostics confirm instability: very large `t_stat_sd` and variance ratios for OLS in these cells (variance-ratio range among checked OLS cells: 9.677 to 12.284).

## 6) Direct answers
- Are extreme OLS RMSE values reproducible from raw results? **Yes.**
- Are they caused by a coding/aggregation bug? **No evidence of a bug.**
- If not a bug, what is the substantive reason? **Finite-sample instability of OLS nuisance estimation when fold-level training size is close to dimensionality (`n_train~=p`), producing outlier `theta_hat`.**
- Should we keep these values in the paper? **Yes, as valid evidence of OLS instability in these design cells.**
