# Major Revision Plan

## Paper: Cross-Fitting, Nuisance Learner Stability, and Finite-Sample Inference in Double Machine Learning

This document defines the revision plan for the resubmission of the paper originally submitted as:

> **Finite-Sample Inference in Double Machine Learning: OLS Instability under Cross-Fitting**

The manuscript was rejected with encouragement to resubmit after substantial revision. The purpose of this file is to control the revision process: paper framing, simulation redesign, code changes, diagnostics, tables/figures, and response-to-reviewer actions.

---

## 1. New Working Title

**Cross-Fitting, Nuisance Learner Stability, and Finite-Sample Inference in Double Machine Learning**

### Rationale

The original title overemphasized OLS instability. The revised title reframes the contribution around a broader and more publishable question: how cross-fitting affects finite-sample inference through fold-level nuisance-estimation stability.

---

## 2. New Core Contribution

The revised paper studies how fold-level nuisance estimation affects finite-sample inference in cross-fitted Double/Debiased Machine Learning (DML).

The main contribution is empirical and diagnostic, not theoretical. The paper evaluates whether fold-level dimensionality and numerical diagnostics help explain coverage distortions, standard-error distortions, and non-convergence across different learners, fold choices, and data-generating processes.

### Main claim to use

> Fold-level dimensionality is a useful empirical diagnostic for finite-sample fragility in cross-fitted DML.

### Main claim to avoid

> (pK/n) is the theoretical determinant of DML reliability.

The revised paper must not overstate the theoretical status of the simulation findings.

---

## 3. Model Framework

The paper will keep the partially linear model (PLM) as the main framework:

[
Y_i = \theta_0 D_i + g_0(X_i) + U_i,
]

where:

* (Y_i) is the outcome,
* (D_i) is the treatment or target regressor,
* (X_i) is the vector of controls/covariates,
* (\theta_0) is the low-dimensional target parameter,
* (g_0(X_i)) is the structural nuisance component,
* (U_i) is the structural error.

The DML nuisance functions are:

[
\ell_0(X_i)=E[Y_i \mid X_i],
\qquad
r_0(X_i)=E[D_i \mid X_i].
]

The orthogonal score is:

[
\psi(W_i;\theta,\eta)
=====================

{Y_i-\ell(X_i)-\theta[D_i-r(X_i)]}[D_i-r(X_i)].
]

The DML estimator solves:

[
\frac{1}{n}\sum_{i=1}^{n}
\psi(W_i;\hat{\theta},\hat{\eta}_{-k(i)})=0.
]

The previous notation error (g_0(X)=E[Y\mid X]) must be removed everywhere.

---

## 4. Fold-Level Dimensionality

For (K)-fold cross-fitting, the nuisance functions are estimated on the training sample:

[
n_{\text{train}} = n\left(1-\frac{1}{K}\right).
]

The main diagnostic ratio is:

[
\rho_{\text{fold}} = \frac{p}{n_{\text{train}}}.
]

This is preferred to vague statements about (pK/n). The paper may mention that (\rho_{\text{fold}}) is closely related to fold-level dimensionality, but the main analysis should use (p/n_{\text{train}}).

---

## 5. Fixed Design Decisions

The following decisions are fixed for the major revision.

### 5.1 Model

* Keep PLM as the main framework.
* Treat the linear model as a special case of PLM.
* Do not replace the PLM by a purely classical linear model.

### 5.2 Dimensionality

The simulation must include:

* low-dimensional regimes,
* moderate-dimensional regimes,
* near-critical regimes,
* high-dimensional / singular regimes.

The paper should not only study (p>n_{\text{train}}), because OLS failure would then be trivial.

### 5.3 Cross-fitting folds

Use:

[
K \in {2,5,10}.
]

### 5.4 Learners

Use the following nuisance learners:

1. OLS
2. Lasso
3. Elastic Net
4. Random Forest
5. Gradient Boosting

Neural networks are optional and should be added only if runtime and tuning are manageable.

### 5.5 Data-generating processes

Use four DGPs:

1. Linear dense independent
2. Linear sparse correlated
3. Nonlinear smooth
4. Threshold / interaction nonsmooth

---

## 6. Proposed Simulation Grid

Initial candidate grid:

[
n \in {200,400,800},
]

[
p \in {25,50,100,150,300},
]

[
K \in {2,5,10}.
]

The final grid should not necessarily include every combination. The selected scenarios should cover values of:

[
\rho_{\text{fold}} = \frac{p}{n_{\text{train}}}
]

approximately in the range:

[
0.05,; 0.15,; 0.30,; 0.60,; 0.90,; 1.10.
]

This gives a clear transition from stable to fragile nuisance-estimation regimes.

---

## 7. DGP Definitions

### DGP 1: Linear Dense Independent

Purpose: baseline linear PLM with independent covariates.

[
X_i \sim N(0,I_p).
]

[
D_i = X_i'\gamma_0 + V_i,
]

[
Y_i = \theta_0D_i + X_i'\beta_0 + U_i.
]

Use dense but controlled coefficients for (\beta_0) and (\gamma_0).

Expected role:

* OLS should perform well when (\rho_{\text{fold}}) is small.
* OLS should become fragile near (\rho_{\text{fold}}\approx 1).
* Penalized learners should be more stable near high-dimensional regimes.

---

### DGP 2: Linear Sparse Correlated

Purpose: sparse signal with correlated covariates.

[
X_i \sim N(0,\Sigma),
]

where (\Sigma) follows an AR(1) structure:

[
\Sigma_{jk} = \rho^{|j-k|}.
]

Recommended:

[
\rho = 0.5.
]

Only a small subset of coefficients in (\beta_0) and (\gamma_0) are nonzero.

Expected role:

* Lasso should perform well under sparsity.
* Elastic Net may be more stable under correlated covariates.
* OLS should become fragile as fold-level dimensionality increases.

---

### DGP 3: Nonlinear Smooth

Purpose: test flexible nuisance learners.

Example:

[
g_0(X_i)
========

\sin(X_{i1}) + X_{i2}^2 + X_{i3}X_{i4}.
]

[
m_0(X_i)
========

0.5X_{i1} + \sin(X_{i2}) + 0.25X_{i3}^2.
]

[
D_i = m_0(X_i)+V_i,
]

[
Y_i = \theta_0D_i + g_0(X_i)+U_i.
]

Expected role:

* RF and Gradient Boosting should be competitive.
* Linear learners may be misspecified.
* This DGP directly addresses the need for nonlinear nuisance functions.

---

### DGP 4: Threshold / Interaction Nonsmooth

Purpose: test nonsmooth nuisance functions.

Example:

[
g_0(X_i)
========

1{X_{i1}>0}
+
X_{i2}1{X_{i3}>0}
+
0.5X_{i4}X_{i5}.
]

[
m_0(X_i)
========

1{X_{i1}+X_{i2}>0}
+
0.5X_{i3}.
]

[
D_i = m_0(X_i)+V_i,
]

[
Y_i = \theta_0D_i + g_0(X_i)+U_i.
]

Expected role:

* Tree-based learners should be useful.
* Linear learners should be less reliable.
* This DGP directly addresses the reviewer request for nonsmooth or threshold designs.

---

## 8. Learner Specifications

### 8.1 OLS

Use ordinary least squares when feasible.

If (p \geq n_{\text{train}}), OLS is singular. The code must handle this explicitly.

Possible rule:

* if OLS matrix is full rank: estimate normally;
* if rank deficient: record failure or use Moore-Penrose pseudoinverse only if clearly documented.

Preferred approach:

* record OLS rank deficiency as part of instability diagnostics;
* do not hide singularity by silently using pseudoinverse.

---

### 8.2 Lasso

Use cross-validated Lasso or a theoretically motivated fixed penalty.

Recommended practical choice:

* use cross-validated Lasso;
* use standardized covariates;
* use fixed random seed;
* record selected penalty and number of selected variables if possible.

---

### 8.3 Elastic Net

Use cross-validated Elastic Net.

Recommended:

* use a small grid over (\alpha), for example:
  [
  \alpha \in {0.1,0.5,0.9}.
  ]
* record selected penalty and mixing parameter if possible.

---

### 8.4 Random Forest

Use Random Forest as a flexible nonlinear learner.

Recommended initial parameters:

* fixed number of trees;
* minimum leaf size controlled;
* random seed fixed;
* no excessive hyperparameter tuning in the main simulation.

If runtime is too high, RF can be run on a reduced grid but should remain in the paper.

---

### 8.5 Gradient Boosting

Use Gradient Boosting as the main flexible nonlinear benchmark.

Recommended:

* HistGradientBoostingRegressor or GradientBoostingRegressor;
* fixed basic tuning;
* record runtime if feasible.

Gradient Boosting is mandatory because it directly addresses the reviewer request for flexible learners.

---

## 9. Diagnostics to Add

For each replication and scenario, record:

### 9.1 Fold-level design diagnostics

* (n_{\text{train}})
* (p)
* (\rho_{\text{fold}} = p/n_{\text{train}})
* rank of the fold-level design matrix
* rank deficiency indicator
* minimum eigenvalue of (X'X/n_{\text{train}})
* condition number of (X'X/n_{\text{train}})

### 9.2 Nuisance prediction diagnostics

For outcome nuisance (\ell_0(X)):

* MSPE for (Y)
* out-of-sample (R^2_Y)

For treatment nuisance (r_0(X)):

* MSPE for (D)
* out-of-sample (R^2_D)

### 9.3 DML inference diagnostics

* estimate (\hat{\theta})
* standard error
* confidence interval lower bound
* confidence interval upper bound
* coverage indicator
* confidence interval length
* standard-error distortion, if oracle or empirical benchmark is available
* convergence or failure indicator

---

## 10. Final Simulation Metrics

For each scenario, report:

* bias
* median bias
* mean absolute bias
* MAE
* RMSE
* empirical standard deviation
* average standard error
* standard-error ratio
* coverage
* average confidence interval length
* non-convergence rate
* average condition number
* share of rank-deficient folds
* average minimum eigenvalue
* nuisance MSPE
* nuisance out-of-sample (R^2)

Mean absolute bias must be included because average bias can hide positive and negative cancellations.

---

## 11. Tables and Figures

The revised paper must reduce redundancy.

### Main tables

1. **Table 1: Simulation Design**

   * DGPs
   * (n,p,K)
   * learners
   * number of replications
   * target parameter

2. **Table 2: Main Simulation Results**

   * grouped by (\rho_{\text{fold}})
   * bias
   * mean absolute bias
   * coverage
   * CI length
   * non-convergence rate

3. **Table 3: Numerical Diagnostics**

   * condition number
   * minimum eigenvalue
   * rank deficiency
   * nuisance (R^2)

4. **Table 4: Empirical Illustration or Stress Test**

   * if empirical section retained

### Main figures

1. Coverage vs. (\rho_{\text{fold}})
2. Standard-error distortion vs. condition number
3. (K=2,5,10) sensitivity plot
4. Learner comparison in nonlinear DGPs

Avoid presenting the same metric repeatedly as a table, bar plot, and line plot.

---

## 12. Empirical Section Plan

The original 401(k) application does not support the instability regime because the sample is large and the number of covariates is small.

Use one of two strategies.

### Option A: Empirical Illustration

Rename the section:

> Empirical Illustration

State clearly:

> This application is not intended to reproduce the high fold-level dimensionality regime studied in the simulations. Instead, it illustrates how the proposed diagnostics can be reported in a standard empirical DML application.

### Option B: Empirical Stress Test

Construct a more relevant empirical design:

* use 401(k) data;
* create polynomial and interaction controls;
* run subsamples:
  [
  n \in {300,500,1000}
  ]
* compare learners across:
  [
  K \in {2,5,10}.
  ]
* report (\rho_{\text{fold}}), diagnostics, and estimates.

Recommended strategy:

* implement Option A first;
* implement Option B if time and runtime permit.

---

## 13. Revised Paper Structure

The revised manuscript should use the following structure.

### 1. Introduction

* Motivate finite-sample implementation problems in DML.
* Explain why cross-fitting changes the nuisance-estimation sample size.
* Reframe contribution around learner stability and diagnostics.
* State that the contribution is simulation-based and diagnostic.

### 2. DML Framework and Cross-Fitting

* Present the PLM.
* Define nuisances correctly:
  [
  \ell_0(X)=E[Y|X],\quad r_0(X)=E[D|X].
  ]
* Present the orthogonal score.
* Explain cross-fitting briefly.
* Avoid long textbook exposition.

### 3. Fold-Level Nuisance Estimation and Diagnostics

* Define (n_{\text{train}}).
* Define (\rho_{\text{fold}}).
* Define instability in statistical/computational terms:

  * rank deficiency,
  * ill-conditioning,
  * standard-error distortion,
  * non-convergence,
  * poor nuisance prediction.

### 4. Simulation Design

* DGPs.
* Learners.
* Fold choices.
* Metrics.
* Replications.
* Runtime/computational notes if needed.

### 5. Simulation Results

* Organize by (\rho_{\text{fold}}), (K), DGP, and learner.
* Focus on coverage and diagnostics.
* Discuss OLS as baseline, not as the only story.
* Compare regularized and nonlinear learners.

### 6. Empirical Illustration

* Use careful wording.
* Do not claim the 401(k) data confirm the instability regime unless a stress test is added.

### 7. Practical Recommendations

Recommend that applied DML papers report:

* number of folds (K),
* fold-level training size,
* (\rho_{\text{fold}}),
* learners used,
* nuisance prediction quality,
* learner sensitivity,
* repeated cross-fitting sensitivity,
* numerical diagnostics when linear learners are used.

### 8. Conclusion

* Summarize findings.
* Emphasize diagnostic contribution.
* Avoid theoretical overclaiming.

---

## 14. Literature Review Additions

The revised literature review should add and explicitly connect the following strands.

### Core DML

* Chernozhukov et al. (2018)
* Ahrens et al. “An Introduction to Double/Debiased Machine Learning”
* Chernozhukov et al., Applied Causal Inference Powered by ML and AI

### Learner choice and tuning

* Bach et al. on DoubleML and hyperparameter tuning
* Bischl et al. on hyperparameter optimization
* Machlanski et al. on model evaluation in causal effect estimation
* Ahrens et al. on model averaging and DML

### Applied DML and implementation

* Fuhr and Papies
* Clarke and Polselli
* Baiardi and Naghi
* Wüthrich and Zhu

The literature review must not merely list papers. Each paper should be connected to the revised contribution.

---

## 15. Code Task List

### Task 1: Add `n_folds`

Add `n_folds` to simulation configuration and all relevant functions.

Required values:

```python
n_folds_grid = [2, 5, 10]
```

---

### Task 2: Fix DGP consistency

Ensure that code and paper describe identical DGPs.

No mismatch is acceptable between:

* paper formulas,
* simulation configuration,
* DGP implementation,
* table labels.

---

### Task 3: Add new DGPs

Implement:

```python
linear_dense_independent
linear_sparse_correlated
nonlinear_smooth
threshold_interaction
```

---

### Task 4: Add learners

Implement or verify:

```python
ols
lasso
elastic_net
random_forest
gradient_boosting
```

---

### Task 5: Add diagnostics

Add per-replication diagnostics:

```python
condition_number
min_eigenvalue
rank
rank_deficient
fold_train_size
fold_ratio
nuisance_mse_y
nuisance_mse_d
nuisance_r2_y
nuisance_r2_d
```

---

### Task 6: Add final metrics

Add aggregation for:

```python
bias
median_bias
mean_absolute_bias
mae
rmse
empirical_sd
average_se
se_ratio
coverage
ci_length
non_convergence_rate
rank_deficiency_rate
average_condition_number
average_min_eigenvalue
```

---

### Task 7: Update tables

Generate:

```text
table_simulation_design
table_main_results
table_diagnostics
table_empirical
```

---

### Task 8: Update figures

Generate:

```text
figure_coverage_by_fold_ratio
figure_condition_number_vs_se_distortion
figure_k_sensitivity
figure_nonlinear_dgp_learners
```

---

### Task 9: Update tests

Add or update tests for:

* DGP formulas;
* fold splitting;
* learner registry;
* metric aggregation;
* diagnostic calculations;
* reproducibility under fixed seed.

---

## 16. Pilot Simulation Plan

Before full simulation, run a pilot.

### Pilot settings

* (R=100) replications.
* All DGPs.
* All learners.
* (K={2,5,10}).
* Reduced (n,p) grid.

### Pilot goals

* detect coding bugs;
* estimate runtime;
* verify that diagnostics behave sensibly;
* check whether results support the revised contribution.

### Pilot decision rule

If results do not show a clear relationship between fold-level diagnostics and inference quality, weaken the claims further and focus on reporting diagnostics rather than asserting a strong pattern.

---

## 17. Full Simulation Plan

Target:

[
R=1000
]

replications per scenario.

If runtime becomes too high:

* run (R=1000) for OLS, Lasso, Elastic Net;
* run (R=500) for RF and Gradient Boosting;
* report the difference transparently.

Preferred final standard:

* same (R) for all learners if computationally feasible.

---

## 18. Response Letter Plan

Prepare a detailed response-to-reviewer document with the following columns:

| Reviewer Comment | Action Taken | Location in Revised Manuscript |
| ---------------- | ------------ | ------------------------------ |

Tone:

* professional;
* non-defensive;
* direct;
* specific.

Use phrases such as:

* “We agree with the reviewer.”
* “We have revised the manuscript accordingly.”
* “We no longer present this claim as a theoretical result.”
* “We added simulations with (K=5) and (K=10).”
* “We added nonlinear and nonsmooth DGPs.”
* “We corrected the PLR notation.”
* “We reframed the empirical application.”

Avoid defensive language.

---

## 19. Final Consistency Checklist

Before resubmission, check:

* [ ] title changed;
* [ ] abstract rewritten;
* [ ] introduction rewritten;
* [ ] PLR notation corrected;
* [ ] no (g_0(X)=E[Y|X]) error remains;
* [ ] (K=2,5,10) included;
* [ ] nonlinear DGP included;
* [ ] nonsmooth DGP included;
* [ ] RF included;
* [ ] Gradient Boosting included;
* [ ] mean absolute bias included;
* [ ] diagnostics included;
* [ ] empirical section reframed;
* [ ] redundant figures removed;
* [ ] figures readable;
* [ ] tables self-contained;
* [ ] references corrected;
* [ ] reviewer response letter completed;
* [ ] code reproduces all tables and figures;
* [ ] random seeds documented;
* [ ] repository README updated.

---

## 20. Current Status

| Phase    | Description                     | Status      |
| -------- | ------------------------------- | ----------- |
| Phase 1  | Revision plan and design freeze | In progress |
| Phase 2  | Exact simulation design table   | Not started |
| Phase 3  | Theory and notation rewrite     | Not started |
| Phase 4  | Code changes                    | Not started |
| Phase 5  | Pilot simulations               | Not started |
| Phase 6  | Full simulations                | Not started |
| Phase 7  | Tables and figures              | Not started |
| Phase 8  | Paper rewrite                   | Not started |
| Phase 9  | Response letter                 | Not started |
| Phase 10 | Final consistency check         | Not started |

---

## 21. Immediate Next Step

After this file is created, the next step is:

> Define the exact simulation design table before editing code.

This includes final choices for:

* DGPs;
* (n,p,K) scenarios;
* learners;
* replications;
* metrics;
* which scenarios are full-grid and which are reduced-grid.
