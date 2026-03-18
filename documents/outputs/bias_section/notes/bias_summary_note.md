# Bias Section Summary Note

## Mean bias by learner
- OLS: +0.001855
- Lasso: +0.016493
- Elastic Net: +0.036437

## Mean absolute bias by learner
- OLS: 0.013093
- Lasso: 0.016493
- Elastic Net: 0.036437

## Maximum absolute bias by learner
- OLS: 0.030843
- Lasso: 0.025836
- Elastic Net: 0.054729

## Scenarios with the largest absolute bias
- Linear Baseline, n200_p150, Elastic Net: bias=+0.054729, abs_bias=0.054729
- Linear Baseline, n200_p100, Elastic Net: bias=+0.047694, abs_bias=0.047694
- Linear Sparse Correlated, n200_p150, Elastic Net: bias=+0.041267, abs_bias=0.041267
- Linear Baseline, n300_p150, Elastic Net: bias=+0.039165, abs_bias=0.039165
- Linear Sparse Correlated, n200_p100, Elastic Net: bias=+0.038904, abs_bias=0.038904
- Linear Baseline, n300_p100, Elastic Net: bias=+0.034508, abs_bias=0.034508

## Bias sign changes across scenarios
- OLS: yes
- Lasso: no
- Elastic Net: no
