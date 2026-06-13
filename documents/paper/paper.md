---
title: "Finite-Sample Inference in Double Machine Learning"
bibliography: refs.bib
format:
  pdf:
    include-in-header: preamble.tex
---

# Introduction

This paper studies finite-sample inference in partially linear Double Machine
Learning (DML) under cross-fitting. The central diagnostic is the fold-level
ratio \(p / n_{\text{train}}\), where \(n_{\text{train}} = n(1 - 1/K)\). The
simulation design isolates how this ratio affects nuisance estimation,
treatment-effect accuracy, and confidence-interval coverage.

# Design

The final design is simulation-only. It uses four DGPs:

- `dense_linear_independent`
- `sparse_linear_independent`
- `sparse_linear_correlated`
- `weak_signal_sparse`

The nuisance learners are OLS, Lasso, Elastic Net, and Gradient Boosting. The
grid uses \(n \in \{250, 500, 1000\}\), \(p \in \{25, 50, 100, 150, 300\}\),
and \(K \in \{2, 5, 10\}\), producing 720 scenarios.

# Estimator

The estimator is the partially linear DML estimator of
@chernozhukov2018dml. For each scenario and replication, nuisance functions are
estimated on training folds and evaluated on held-out folds. The treatment
effect is then estimated from residualized outcome and treatment variables.

# Outputs

The workflow produces:

- simulation summaries in `documents/outputs/aggregated/`
- publication tables in `documents/outputs/tables/`
- diagnostic figures in `documents/outputs/figures/`

# References
