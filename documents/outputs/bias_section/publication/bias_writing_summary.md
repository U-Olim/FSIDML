# Bias Writing Summary

## 1) Mean bias by learner
- OLS: +0.001855
- Lasso: +0.016493
- Elastic Net: +0.036437

## 2) Max and min bias by learner
- OLS: max=+0.030843, min=-0.028702
- Lasso: max=+0.025836, min=+0.007545
- Elastic Net: max=+0.054729, min=+0.024136

## 3) Mean absolute bias by learner
- OLS: 0.013093
- Lasso: 0.016493
- Elastic Net: 0.036437

## 4) By DGP: largest positive and largest negative bias scenarios
- Linear Baseline: largest positive = n200_p150 (Elastic Net, +0.054729); largest negative = n200_p100 (OLS, -0.028702)
- Linear Sparse Correlated: largest positive = n200_p150 (Elastic Net, +0.041267); largest negative = n400_p100 (OLS, -0.004301)

## 5) Sign changes across scenarios by learner
- OLS: yes
- Lasso: no
- Elastic Net: no

## 6) Visible patterns
- Combined bar charts: Elastic Net bars are consistently positive and typically highest; OLS bars sit closest to zero with occasional negatives.
- Combined line graphs: learner ordering is stable across scenarios (Elastic Net > Lasso > OLS), and OLS is the only learner with frequent sign switches.
- Combined heatmaps: the largest positive cells cluster in Elastic Net, while OLS shows mixed sign cells and lower absolute intensity.
