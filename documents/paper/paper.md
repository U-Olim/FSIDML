---
bibliography: refs.bib
citeproc: true
format:
  html:
    from: markdown+citations
    css: style.css
    toc: false
    number-sections: false
  pdf:
    from: markdown+citations
    toc: false
    number-sections: false
    fontsize: 11pt
    geometry:
      - margin=1in
    linestretch: 1.15
    pdf-engine: xelatex
    include-in-header: preamble.tex
---

::: {=latex}
\begin{titlepage}
\thispagestyle{empty}
\begin{center}

\vspace*{1.2cm}

{\fontsize{16pt}{19pt}\selectfont\bfseries Finite-Sample Inference in Double Machine Learning: OLS Instability under Cross-Fitting\par}

\vspace{1.2cm}

{\fontsize{12pt}{14pt}\selectfont Olimjon Umurzokov\footnotemark[1], Umidjon Kostaev\footnotemark[2]\par}

\vspace{1.2cm}

{\fontsize{11pt}{13pt}\selectfont March 2026\par}

\end{center}

\vspace{0.6cm}
{\fontsize{11pt}{13pt}\selectfont\textbf{Abstract}\par}
\vspace{0.10cm}
{\fontsize{10pt}{12pt}\selectfont This paper studies the finite-sample reliability of Double Machine Learning (DML) in settings with moderate dimensionality ($p < n$). While theory focuses on $p/n$, DML with K-fold cross-fitting uses only $n/K$ observations in each fold. This makes $pK/n$ the relevant measure of dimensionality.

We show that when $pK/n \approx 1$, ordinary least squares (OLS) nuisance estimation becomes ill-conditioned within folds, leading to instability in the DML procedure. This instability does not change the main estimate much, but it causes the standard errors to be severely inaccurate. As a result, confidence intervals can display large undercoverage.

We compare OLS with regularized estimators, including Lasso and Elastic Net, in partially linear models across a range of Monte Carlo designs. Regularization avoids this instability by making nuisance estimates more stable and reducing variance, leading to significant improvements in coverage. However, coverage remains below nominal levels, reflecting residual finite-sample errors.

An empirical application to the 401(k) eligibility data confirms these patterns in a low-dimensional setting. Overall, the results show that finite-sample reliability of DML is controlled by fold-level dimensionality, and that regularization improves but does not fully resolve inference problems in such settings.\par}
\vspace{0.20cm}
{\fontsize{10pt}{12pt}\selectfont\textbf{Keywords:} Double Machine Learning; Finite-sample inference; Monte Carlo simulation; Nuisance estimation; OLS; Lasso; Elastic Net; Cross-Fitting\par}
\footnotetext[1]{University of Bonn; s04oumur@uni-bonn.de}
\footnotetext[2]{Tashkent State University of Economics; umidjonkostaev2025@gmail.com}
\end{titlepage}
\clearpage
\setcounter{page}{1}
:::

## 1. Introduction

Double Machine Learning (DML) estimates causal effects while using machine learning methods to flexibly model nuisance functions. Its theory is based on large-sample results that do not focus on the detailed structure of cross-fitting. In practice, however, cross-fitting with $K$ folds reduces the effective training sample in each fold to approximately $n/K$ observations. When the number of covariates $p$ approaches $n/K$, unregularized nuisance estimation becomes poorly conditioned at the fold level.

This paper shows that the relevant quantity for finite-sample reliability is the fold-level ratio $p/(n/K)$, equivalently $pK/n$, rather than the global ratio $p/n$ emphasized in standard asymptotic theory. This distinction is empirically relevant. For example, a dataset with $n = 200$ and $p = 100$ is typically considered moderate-dimensional since $p < n$. However, with $K = 2$ folds, each nuisance model is estimated on roughly 100 observations, placing OLS at the boundary of singularity. Standard asymptotic guarantees do not cover this regime, as they assume settings where fold-level dimensionality remains well-behaved.

We study how the choice of nuisance estimator affects the finite-sample behavior of DML in partially linear models. The key finding is that instability from OLS-based nuisance estimation primarily affects inference through standard errors rather than through bias. The treatment effect estimates remain approximately centered, but the estimated standard errors become inaccurate. As a result, the problem is not visible when attention is restricted to point estimates alone.

We compare OLS, Lasso, and Elastic Net as nuisance estimators across Monte Carlo designs with moderate dimensionality ($p < n$) under both independent and correlated covariates. The paper makes three contributions. First, we identify the fold-level ratio $pK/n$ as the key determinant of instability in DML. Second, we show that this instability affects inference through standard errors rather than bias, making it difficult to detect in standard practice. Third, we estimate the severity of this problem in detail and show that, when $pK/n \approx 1$, OLS-based DML shows significant undercoverage.

Regularized estimators reduce this issue by stabilizing nuisance estimation in near-singular settings. Although they introduce some bias in the nuisance functions, they decrease the variance of nuisance predictions and lead to better-calibrated standard errors for the treatment effect estimator.

An empirical application to the 401(k) eligibility data confirms these findings in a low-dimensional setting where instability is not expected. Even in this case, regularized nuisance estimators produce more stable standard errors and improved inference across specifications, indicating that the issue is not restricted to simulation designs.

## 2. Literature Review

Recent research in econometrics has increasingly focused on combining machine learning methods with causal inference. These approaches aim to estimate causal parameters while allowing flexible modeling of complex relationships between variables. A central contribution in this area is the Double Machine Learning (DML) framework (Chernozhukov et al., 2018). In this work, the authors propose a procedure that separates the estimation of nuisance functions from the estimation of the causal parameter of interest. By using orthogonal estimating equations and cross-fitting, the method reduces the impact of errors in nuisance estimation and allows valid inference for low-dimensional causal parameters.

The work by Chernozhukov et al. (2024) provides an overview of how machine learning methods can be integrated into causal inference applications. These methods are particularly useful when the model includes many control variables or when flexible functional forms are required to capture complex relationships in the data.

Several studies have examined the role of regularization methods in treatment-effect estimation when many potential control variables are available. A key contribution is provided in Belloni, Chernozhukov, and Hansen (2014). This work develops a framework for valid inference after variable selection using the Lasso estimator. Their results show that regularization can be used to select relevant controls while preserving reliable inference for the treatment effect. Related work in Farrell (2015) studies robust inference procedures for treatment effects when the number of covariates is large relative to the sample size.

The theoretical foundation for regularization methods used in these settings is discussed extensively in Buhlmann and van de Geer (2011). This work provides a comprehensive treatment of penalized regression methods such as Lasso and Elastic Net and analyzes their statistical properties. These methods have become standard tools for estimating models that include many potential predictors, particularly when traditional estimation methods may become unstable.

Also, the practical performance of machine learning methods for causal inference through simulation studies has been examined by Naghi (2021). In this work, authors conduct a systematic evaluation of several causal machine learning approaches and compares their behavior across different data generating processes. Such studies highlight that the empirical performance of causal estimators may vary a lot depending on modeling choices and implementation details. In particular, simulation analyses provide useful insights into the stability and reliability of treatment effect estimators.

More recent work by Ballinari and Bearth (2024) investigates how modifications of the standard DML framework can improve performance in finite samples. Their results show that implementation choices and estimation strategies can have a meaningful impact on the behavior of DML estimators. This research emphasizes that, beyond theoretical guarantees, careful empirical evaluation remains important for understanding how these methods perform in practice.

The role of implementation choices in Double Machine Learning has been studied in the context of tuning procedures. Bach et al. (2024) analyze how hyperparameter selection affects the performance of DML estimators and demonstrate that tuning decisions may influence both estimation accuracy and inference. Their findings underline that the practical performance of DML depends not only on the underlying framework but also on the methods used to estimate nuisance components.

## 3. Model and Simulation Design

### 3.1 Model

We consider a partially linear regression model for estimating the causal effect of a treatment variable on an outcome in the presence of observed covariates. Let the observed data be an independent and identically distributed sample
$$
W_i = (Y_i, D_i, X_i), \quad i = 1,\dots,n
$$
where $Y_i \in \mathbb{R}$ is the outcome variable, $D_i \in \mathbb{R}$ is the treatment variable, and $X_i \in \mathbb{R}^{p}$ is a $p$-dimensional vector of covariates.

Then the relationship between the variables is described by the **Partially Linear Model:**
$$
Y_i = \theta_0 D_i + g_0(X_i) + U_i, \qquad D_i = m_0(X_i) + V_i
$$
where
\begin{itemize}
\item $\theta_0$ is the parameter of interest representing the causal effect of the treatment $D_i$ on the outcome $Y_i$,
\item $g_0(X_i)$ is an unknown regression function capturing the effect of covariates $X_i$,
\item $U_i$ is an error term.
\item $m_0(X_i)$ denotes the conditional expectation of the treatment given covariates and $V_i$ represents the residual variation in the treatment.
\end{itemize}

\textbf{Orthogonality Conditions.}
Identification of the parameter $\theta_0$ relies on the following conditional moment assumptions
$$
E[U_i \mid X_i, D_i] = 0, \qquad E[V_i \mid X_i] = 0
$$
These assumptions imply that the error components are orthogonal to the regressors after conditioning on the appropriate variables.

\textbf{Conditional Expectation Functions.}
Define the nuisance functions
$$
g_0(X_i) = E[Y_i \mid X_i], \qquad m_0(X_i) = E[D_i \mid X_i]
$$
Using these functions, we define the residualized variables
$$
\tilde{Y}_i = Y_i - g_0(X_i), \qquad \tilde{D}_i = D_i - m_0(X_i).
$$
Substituting these expressions into the structural equation yields
$$
\tilde{Y}_i = \theta_0 \tilde{D}_i + U_i.
$$

\textbf{Estimator.}
The parameter $\theta_0$ is estimated by solving the sample analogue of the moment condition
$$
\frac{1}{n}\sum_{i=1}^{n}
(D_i - \hat m(X_i))(Y_i - \hat g(X_i) - \hat{\theta}D_i) = 0.
$$
Solving this equation for $\hat{\theta}$ yields
$$
\hat{\theta} =
\frac{
\frac{1}{n}\sum_{i=1}^{n}
(D_i - \hat m(X_i))(Y_i - \hat g(X_i))
}{
\frac{1}{n}\sum_{i=1}^{n}
(D_i - \hat m(X_i))D_i
}.
$$

\textbf{Orthogonal Moment Condition.}
The parameter $\theta_0$ can also be identified through the orthogonal moment condition
$$
E \left[(D_i - m_0(X_i))(Y_i - g_0(X_i) - \theta_0 D_i)\right] = 0.
$$
Based on this moment condition, define the score function
$$
\psi(W_i; \theta, \eta) =
(D_i - m(X_i))(Y_i - g(X_i) - \theta D_i),
$$
where $\eta = (g,m)$ denotes the nuisance parameters.

To reduce the risk of overfitting when estimating the nuisance functions, we implement cross-fitting. The sample is partitioned into $K$ folds, and nuisance functions are estimated on one part of the data while predictions are generated for observations in the held-out fold.

In the simulation study we use $K = 2.$ For each fold, the nuisance functions $\hat g(X)$ and $\hat m(X)$ are estimated using the training subsample and then applied to compute predictions for the validation subsample. The residualized variables are constructed using these predictions, and the final estimator is computed by combining the results across folds.

### 3.2 Ill-Conditioning Fold-Level Mechanism

The instability observed in our simulations arises from the interaction between cross-fitting and the dimensionality of the covariates. In DML with $K$-fold cross-fitting, each nuisance model is estimated on a subsample of size approximately $n_{\text{fold}} = n/K$. As a result, the relevant dimensionality is not determined by the ratio $p/n$, but by the fold-level ratio
$$
\frac{p}{n_{\text{fold}}} = \frac{pK}{n}.
$$

To see the implications, consider the OLS estimator used for nuisance estimation within each fold:
$$
\hat{\beta}^{(k)} = (X_k^\top X_k)^{-1} X_k^\top Y_k,
$$
where $X_k$ denotes the design matrix in fold $k$. The finite-sample behavior of $\hat{\beta}^{(k)}$ depends critically on the conditioning of the Gram matrix $X_k^\top X_k$. When $p$ approaches $n_{\text{fold}}$, the matrix $X_k^\top X_k$ becomes ill-conditioned, with its smallest eigenvalue approaching zero. In this regime, the inverse $(X_k^\top X_k)^{-1}$ becomes unstable, leading to large variance in the estimated coefficients:
$$
\mathrm{Var}(\hat{\beta}^{(k)}) \propto (X_k^\top X_k)^{-1}.
$$

This ill-conditioning is not an extreme case but arises naturally in our simulation designs. For example, with $(n,p) = (200,100)$ and $K=2$, each fold contains roughly 100 observations and 100 covariates, so that $p/n_{\text{fold}} \approx 1$. A similar situation occurs for $(n,p) = (300,150)$. Although these settings satisfy $p < n$ at the full-sample level, nuisance estimation is effectively performed at the boundary of matrix invertibility within each fold.

Importantly, this mechanism explains why DML can exhibit unstable inference even when standard asymptotic conditions appear to be satisfied. The orthogonality property of DML ensures that small estimation errors in the nuisance functions do not induce first-order bias in the treatment effect estimator. However, orthogonality does not control the variance of the nuisance estimates. When nuisance estimation is unstable due to ill-conditioning, the resulting predictions $\hat{g}(X)$ and $\hat{m}(X)$ become highly variable. This variability spreads to the orthogonalized residuals and inflates the variance of the DML score, leading to inaccurate standard errors.

As a consequence, the breakdown of inference documented in our simulations is driven by variance inflation rather than bias. The treatment effect estimates remain approximately centered, but confidence intervals exhibit considerable undercoverage. This also explains why the problem is difficult to detect in practice: analyses that focus only on point estimates may not reveal the instability.

Overall, the results indicate that the threshold for instability in DML with OLS nuisance estimation is determined by the fold-level ratio $pK/n$, rather than the ratio $p/n$. When $pK/n \approx 1$, the nuisance estimation problem becomes ill-conditioned, and standard inference procedures can fail even in settings that are conventionally viewed as moderate-dimensional.

### 3.3 Data Generating Processes

A data generating process (DGP) describes how the data used in a simulation are created. It specifies the underlying model that links the variables together and defines the probability distributions of the random components. The covariate vector $X_i \in \mathbb{R}^p$ is generated from a multivariate normal distribution $X_i \sim N(0,\Sigma).$ Two covariance structures are considered. In the baseline setting the covariates are independent, implying $\Sigma = I_p,$ where $I_p$ denotes the $p \times p$ identity matrix. In the correlated setting the covariance matrix follows the autoregressive structure
$$
\Sigma_{jk} = \rho^{|j-k|},
$$
where $\rho = 0.5$. This specification introduces moderate correlation among the covariates and creates a more challenging estimation environment.

The signal in the model is generated through a sparse linear combination of covariates 
$$S_i = \sum_{j=1}^{s} \beta_j X_{ij}.$$
The coefficients are defined as $\beta_j \propto \frac{1}{j^2},$ and normalized to satisfy $\|\beta\|_2 = 1.$ The sparsity level is defined as $s = \min(5,p),$ so that only a small subset of covariates contributes to the signal.

\textbf{Noise Terms.}
The stochastic disturbances are generated as independent Gaussian noise 
$$
U_i \sim N(0,1), \qquad V_i \sim N(0,1).
$$
These error terms represent unobserved factors affecting the outcome and treatment variables.

\paragraph{DGP Specifications}
Based on the framework described above, the simulation study considers two data generating processes.

\textbf{DGP 1: Linear Baseline.}
The first design represents a baseline scenario with independent covariates. In this case the covariance matrix satisfies $\Sigma = I_p.$ The structural relationships between the variables remain linear and the nuisance functions follow the sparse signal structure described above. This setting serves as a benchmark environment for evaluating the finite-sample behavior of the estimator.

\textbf{DGP 2: Linear Sparse Correlated.}
The second design introduces correlation among the covariates through the autoregressive covariance structure $\Sigma_{jk} = \rho^{|j-k|}, \quad \rho = 0.5.$ All other components of the data generating process remain unchanged. The presence of correlated covariates increases the complexity of the estimation problem and allows us to evaluate the robustness of the estimator under more realistic dependence structures among the control variables.
Taken together, these two DGPs allow us to compare estimator performance in a baseline environment with independent controls and in a more challenging setting characterized by correlated covariates.

### 3.4 Simulation Design

In DML with $K$-fold cross-fitting, nuisance functions are estimated on subsamples of size $n/K$. As a result, the relevant measure of dimensionality is the fold-level ratio $pK/n$, rather than the ratio $p/n$. The simulation design is constructed to vary $pK/n$ and study how it affects the stability of nuisance estimation and the resulting inference.

Also, the simulation framework provides a controlled environment in which the true parameter $\theta_0$ is known. Since the data generating process is fully specified, this allows for a direct evaluation of estimation accuracy across repeated samples.

| Design Component                   | Specification                             |
| ---------------------------------- | ----------------------------------------- |
| Data-generating processes (DGPs)   | linear_baseline; linear_sparse_correlated |
| Sample sizes ($n$)                 | 200; 300; 400                             |
| Covariate dimensions ($p$)         | 100; 150                                  |
| Nuisance estimators                | OLS; Lasso; Elastic Net                   |
| True treatment effect ($\theta_0$) | 1.0                                       |
| Cross-fitting folds                | 2                                         |
| Base random seed                   | 123                                       |
| Replications per scenario          | 1000                                      |
| Total scenario count               | $3 \times 2 \times 2 \times 3 = 36$       |

A key feature of the DML framework is that the estimation of the parameter of interest depends on the estimation of nuisance functions. In the partially linear model, the nuisance functions are defined as
$$
g_0(X) = E[Y \mid X], \qquad m_0(X) = E[D \mid X].
$$
These functions are unknown in practice and must be estimated from the data. In the simulation study, we compare three commonly used regression methods for estimating these nuisance components: OLS, Lasso and Elastic Net.

\textbf{\textit{Ordinary Least Squares (OLS).}}
The first estimator is the standard linear regression estimator. OLS estimates the nuisance functions by fitting linear models of the form
$$
\hat g(X_i) = X_i'\hat{\beta}, \qquad \hat m(X_i) = X_i'\hat{\gamma}.
$$
OLS serves as a natural benchmark estimator because it represents the classical approach to modeling conditional expectations in regression analysis.

\textbf{\textit{Lasso.}}
The second estimator is the Lasso regression estimator, which introduces $\ell_1$ regularization to encourage sparsity in the estimated coefficient vector. The Lasso estimator solves the optimization problem
$$
\hat{\beta}_{lasso} =
\arg\min_{\beta}
\left[
\frac{1}{n}\sum_{i=1}^n (Y_i - X_i'\beta)^2
+ \lambda \|\beta\|_1
\right],
$$
where $\lambda$ is a regularization parameter controlling the strength of shrinkage. Lasso is particularly useful when the true model depends only on a small subset of covariates.

\textbf{\textit{Elastic Net.}}
The third estimator considered is Elastic Net regression, which combines $\ell_1$ and $\ell_2$ penalties. The Elastic Net estimator solves
$$
\hat{\beta}_{EN} =
\arg\min_{\beta}
\left[
\frac{1}{n}\sum_{i=1}^n (Y_i - X_i'\beta)^2
+ \lambda_1 \|\beta\|_1
+ \lambda_2 \|\beta\|_2^2
\right].
$$
Elastic Net is designed to perform well in settings where covariates exhibit correlation, as the additional $\ell_2$ penalty stabilizes the estimation relative to pure Lasso regression.

### 3.5. Monte Carlo Design

The simulation study is conducted using a Monte Carlo framework with repeated random sampling. For each replication of the simulation, a dataset of size $n$ is generated according to one of the data generating processes described in Section 3.3. The DML estimation procedure is then applied to the simulated data to obtain an estimate $\hat{\theta}$ of the treatment effect.

This procedure is repeated across a large number of independent simulation replications. In the simulation study we use $R = 1000$ Monte Carlo replications for each experimental setting. Repeating the experiment many times allows us to approximate the sampling distribution of the estimator.

\paragraph{Performance Metrics}
The statistical performance of the estimator is evaluated by several standard metrics commonly used in Monte Carlo simulation studies.

\textbf{Bias.}
Bias measures the systematic deviation of an estimator from the true parameter value. Let $\theta_0$ denote the true value of the treatment effect and let $\hat{\theta}_r$ denote the estimate obtained in simulation replication $r$, where $r = 1, \dots, R$. The theoretical definition of bias is
$$
Bias(\hat{\theta}) = E[\hat{\theta}] - \theta_0.
$$
Since the expectation is unknown, it is approximated using the Monte Carlo average across $R$ replications. The empirical bias is therefore computed as
$$
\widehat{Bias}
=
\frac{1}{R}
\sum_{r=1}^{R}
(\hat{\theta}_r - \theta_0).
$$

\textbf{Root Mean Squared Error (RMSE).}
The RMSE measures the square root of the expected squared deviation between the estimator and the true parameter value. It combines both the bias and the variance of the estimator and therefore provides a comprehensive measure of estimation accuracy. Formally, the RMSE is defined as
$$
RMSE(\hat{\theta}) =
\sqrt{
E[(\hat{\theta} - \theta_0)^2]
}.
$$
In a Monte Carlo simulation, the expectation is approximated using the empirical distribution of the estimator across $r$ replications. The RMSE is therefore estimated as
$$
\widehat{RMSE}
=
\sqrt{
\frac{1}{R}
\sum_{r=1}^{R}
(\hat{\theta}_r - \theta_0)^2
}.
$$
Lower RMSE values indicate more accurate estimators, while larger RMSE values indicate greater estimation error. By comparing RMSE across different nuisance estimation methods, we can assess how the choice of estimator for the nuisance functions affects the overall performance of the DML estimator.

\textbf{Confidence Interval Coverage.}
The confidence intervals are constructed using the asymptotic normal approximation of the standardized estimator. Under regularity conditions, the estimator satisfies
$$
Z_r =
\frac{\hat{\theta}_r - \theta_0}{\hat{\sigma}_r}
\approx N(0,1),
$$
where $\hat{\sigma}_r$ denotes the estimated standard error of $\hat{\theta}_r$ in replication $r$.
A $95\%$ confidence interval for the treatment effect parameter is constructed as
$$
CI_r =
\left[
\hat{\theta}_r - 1.96\,\hat{\sigma}_r,
\;
\hat{\theta}_r + 1.96\,\hat{\sigma}_r
\right].
$$
Coverage measures how often the true parameter value $\theta_0$ lies within the constructed confidence intervals across the Monte Carlo replications. Formally, coverage is defined as
$$
Coverage
=
\frac{1}{R}
\sum_{r=1}^{R}
\mathbf{1}
(\theta_0 \in CI_r),
$$
where $\mathbf{1}(\cdot)$ denotes the indicator function. The indicator function is defined as
$$
\mathbf{1}(\theta_0 \in CI_r)
=
\begin{cases}
1, & \text{if } \theta_0 \text{ lies within the interval } CI_r, \\
0, & \text{otherwise}.
\end{cases}
$$
For a correctly calibrated inference procedure, coverage should be close to the nominal level of $0.95$. If the coverage falls below the nominal level of 0.95, the confidence intervals are too narrow, indicating that the standard errors are underestimated. Conversely, coverage above 0.95 implies that the intervals are too wide, reflecting overestimation of the standard errors.

\textbf{t-Statistic Diagnostics.}
To further evaluate the reliability of statistical inference, we examine the distribution of the standardized estimator using the t-statistic. For each simulation replication $r$, the t-statistic is defined as
$$
t_r =
\frac{\hat{\theta}_r - \theta_0}{\hat{\sigma}_r},
$$
where $\hat{\theta}_r$ is the estimated treatment effect in replication $r$, $\theta_0$ is the true parameter value, and $\hat{\sigma}_r$ is the estimated standard error.

Under correct inference, the standardized estimator should approximately follow a standard normal distribution $t_r \approx N(0,1).$ To evaluate this property, we compute the Monte Carlo mean and standard deviation of the t-statistics. The mean t-statistic is defined as
$$
Mean(t) =
\frac{1}{R}
\sum_{r=1}^{R}
t_r,
$$
and the standard deviation of the t-statistics is
$$SD(t) =
\sqrt{
\frac{1}{R-1}
\sum_{r=1}^{R}
(t_r - \bar{t})^2
},
$$
For a well-calibrated inference procedure, the mean of the t-statistics should be close to zero and their standard deviation should be close to one. Deviations from these values indicate problems with the estimated standard errors or the distributional approximation used for inference.

## 4. Results

This section reports the Monte Carlo results described in Section 3. We compare the performance of the Double Machine Learning estimator under different nuisance estimation strategies—OLS, Lasso, and Elastic Net—using bias, RMSE, and confidence interval coverage as evaluation metrics. Results are presented by metric to facilitate comparison across estimators: first bias, then RMSE and coverage, followed by an analysis of the t-statistic distribution to assess the reliability of the estimated standard errors.

### 4.1 Bias

Bias reflects the average deviation of the estimated treatment effect from the true parameter $\theta_0 = 1$. Across all scenarios the bias remains small, ranging from approximately -0.029 to 0.055 depending on the learner and simulation setting.

::: {=latex}
\setcounter{table}{0}
\input{../outputs/bias_section/publication/table_bias_main_compact.tex}
:::

Table 1 summarizes the overall magnitude of bias across all Monte Carlo scenarios. OLS produces the smallest average bias, with a mean value of approximately 0.0019, which is essentially zero relative to the true parameter value $\theta_0 = 1$. However, the minimum and maximum values show that OLS bias varies between -0.0287 and 0.0308, indicating that its sign changes across scenarios. The near-zero bias of OLS is consistent with the absence of regularization, as the estimator does not shrink coefficients and therefore preserves unbiasedness in finite samples when the model is correctly specified.

In contrast, the regularized estimators exhibit systematically positive bias. Lasso has an average bias of 0.0165, while Elastic Net shows a somewhat larger average bias of 0.0364. The maximum bias observed for Elastic Net reaches 0.0547. In general, the table indicates that systematic estimation error remains modest for all learners. Positive bias observed for Lasso and Elastic Net reflects the effect of shrinkage imposed by regularization. By penalizing large coefficients, these estimators systematically pull estimates toward zero, leading to upward bias in the estimated treatment effect.

::: {=latex}
\input{../outputs/bias_section/publication/include_fig_bias_combined_bars.tex}
:::

The bar charts visualize the bias across scenarios, where the bars remain concentrated within the interval -0.03 to 0.05, confirming that systematic estimation error is limited in magnitude. The figure also illustrates a clear ranking across learners. OLS bars frequently lie close to the zero line and occasionally switch sign, while Lasso and Elastic Net remain consistently above zero.

The line graphs below emphasize how bias evolves across scenarios. As discussed earlier, OLS stays closest to zero and changes sign. Lasso and Elastic Net remain entirely above the zero axis across all six scenarios. These patterns suggest that increasing the sample size from $n=200$ to $n=400$ or increasing the number of covariates from $p=100$ to $p=150$ does not substantially alter the magnitude of bias.

::: {=latex}
\input{../outputs/bias_section/publication/include_fig_bias_combined_lines.tex}
:::

Overall, the simulation results show that systematic estimation error remains limited across all scenarios. OLS produces the smallest average bias and fluctuates around zero, while Lasso and Elastic Net introduce a modest positive bias due to regularization. However, the magnitude of these deviations remains small relative to the true parameter value. These results suggest that bias is not the primary driver of estimator performance in this setting.

### 4.2 Root Mean Squared Error

RMSE combines both bias and dispersion and therefore provides a broader measure of finite-sample accuracy than bias.

::: {=latex}
\input{../outputs/rmse_section/tables/rmse_table_by_dgp.tex}
:::

Table 2 below reports that Lasso achieves the lowest RMSE in every scenario ranging from 0.054 to 0.085, while Elastic Net performs similarly but remains slightly less accurate, between 0.059-0.096. In contrast, OLS displays considerably larger RMSE in several cells, most notably when $(n,p)=(200,100)$ and $(300,150)$, where the RMSE rises to approximately 0.74–0.79, an order of magnitude larger than the values obtained with the regularized learners. In the remaining scenarios, OLS RMSE falls to the range 0.08–0.12, yet it still remains above the corresponding Lasso and Elastic Net values. The same pattern appears in both the Linear Baseline and the Linear Sparse Correlated DGP, indicating that the instability of OLS is not driven by the correlation structure of the controls but rather by the dimensionality of the covariate set relative to the effective training sample.

::: {=latex}
\begin{figure}[!htbp]
\centering
\includegraphics[width=0.90\linewidth]{../outputs/checks/fig_theta_distribution_instability.png}
\caption{Theta-hat density comparison for instability scenarios $(n,p)=(200,100)$ and $(300,150)$ across OLS, Lasso, and Elastic Net.}
\label{fig:theta-distribution-instability}
\end{figure}
:::

Figure 3 clearly reveals the mechanism behind the large RMSE values observed for OLS. In both instability scenarios, the OLS distribution of $\hat{\theta}$ is much wider than the distributions produced by Lasso and Elastic Net. The center of the OLS distribution remains close to the true treatment effect, so the problem is not systematic displacement of the estimator. The problem is that OLS generates a non-negligible number of extreme realizations on both sides of the truth, which creates heavy tails. By contrast, the regularized learners are tightly concentrated around $\theta_0 = 1$, with sharply peaked densities and very thin tails. This means that the main advantage of Lasso and Elastic Net in these scenarios is not that they correct a large bias, but that they prevent rare but very large errors.

Moreover, in most replications, OLS produces estimates close to the true parameter. However, in some runs it becomes unstable and generates very large errors. These extreme outcomes explain the large RMSE values in Table 2. The fact that the same pattern appears in both scenarios suggests that the mechanism is structural, not accidental: once the fold-level training sample becomes too close to the number of controls, unregularized nuisance estimation becomes unstable. The figure therefore identifies the core finite-sample problem in a precise way. OLS does not fail because it is centered incorrectly on average; it fails because its distribution becomes too dispersed and too heavy-tailed to support reliable inference.

::: {=latex}
\begin{figure}[!htbp]
\centering
\includegraphics[width=0.92\linewidth]{../outputs/checks/fig_theta_hat_boxplots_by_learner.png}
\caption{Theta-hat boxplots by learner for instability scenarios.}
\label{fig:theta-hat-boxplots}
\end{figure}
:::

Figure 4 reinforces the message of the density plots and makes the source of the RMSE differences even more visible. The boxplots show that the dispersion of the OLS estimates is dramatically larger than that of the regularized learners. While the median of the OLS estimator remains close to the true treatment effect, the interquartile range is much wider and the number of extreme realizations is significantly higher. In several scenarios, the OLS distribution contains many observations far away from the center, indicating that the estimator occasionally produces very large deviations from the true parameter.

By contrast, the distributions of Lasso and Elastic Net are tightly concentrated around $\theta_0 = 1$. Their interquartile ranges are narrow and extreme values are rare, which implies that the DML estimator remains stable across replications when regularized nuisance learners are used. It means that estimators have similar medians, but OLS shows much greater dispersion.

OLS does not systematically miss the true treatment effect, but it occasionally generates very large errors that considerably increase overall estimation risk. Regularized learners prevent these extreme deviations and therefore produce much more reliable finite-sample performance.

To sum up, the RMSE evidence shows that the main weakness of OLS in this setting is not systematic bias but instability. While the estimator is centered near the true treatment effect on average, it occasionally produces very large errors that inflate the overall estimation risk. Regularized learners avoid these extreme realizations and therefore deliver much more stable treatment effect estimates across replications. As a result, Lasso and Elastic Net achieve significantly lower RMSE even though their average bias is not smaller than OLS.

### 4.3 Coverage

This section presents the coverage of the 95% confidence intervals for the treatment effect estimator across all simulation scenarios. The goal is to evaluate how the choice of nuisance estimator affects the reliability of statistical inference in finite samples.

::: {=latex}
\input{../outputs/coverage_section/tables/coverage_table_by_dgp.tex}
:::

Table 3 reveals a clear pattern: the coverage of OLS is considerably below the nominal level in most scenarios and becomes extremely low in the most demanding cells. In particular, when $(n,p)=(200,100)$ and $(300,150)$ the coverage drops to around 0.15–0.19 in both data-generating processes, indicating a severe breakdown of inference reliability. Even in scenarios where OLS performs better, the coverage remains well below the nominal level of 0.95.

On the other hand, the regularized learners significantly improve inference relative to OLS, but their coverage remains below the nominal 95% level in all scenarios. Lasso consistently delivers coverage rates much closer to the nominal level than OLS, typically between 0.89 and 0.93 across scenarios. Elastic Net performs similarly but tends to produce slightly lower coverage than Lasso. Importantly, the same pattern appears in both the linear_baseline and the linear_sparse_correlated DGP, indicating that the difference across learners is not driven by the correlation structure of the controls.

::: {=latex}
\begin{figure}[!htbp]
\centering
\includegraphics[width=0.95\linewidth]{../outputs/coverage_section/figures/fig_coverage_grouped_bar.png}
\caption{Coverage by scenario and learner with nominal 95\% reference line.}
\label{fig:coverage-grouped-bar}
\end{figure}
:::

Figure 5 visualizes the coverage results across scenarios and reinforces the patterns observed in Table 3. The horizontal reference line marks the nominal 95% coverage level, which provides a clear benchmark for evaluating the reliability of the different nuisance estimators. Across both data-generating processes, the bars corresponding to OLS lie well below this reference line in most scenarios, indicating systematic undercoverage. In contrast, the bars for Lasso and Elastic Net remain consistently close to the nominal level across scenarios.

The severe undercoverage observed in several scenarios suggests that the estimated standard errors do not adequately reflect the true variability of the estimator. When the coverage rate collapses to values around 0.15–0.19, this indicates that the confidence intervals are far too narrow relative to the actual dispersion of the estimates.

### 4.4  $t$-statistics

We now examine the standardized estimator to evaluate whether the distributional assumptions underlying inference are satisfied. If the procedure is valid, the resulting $t$-statistics should follow an approximate $\mathcal{N}(0,1)$ distribution. Deviations from this benchmark provide direct evidence on whether errors arise from bias, incorrect scaling, or excess dispersion.

Table A2 reports diagnostic statistics for the t-statistics of the treatment effect estimator. The results reveal a striking difference across nuisance estimators. For OLS, the standard deviation of the t-statistics becomes extremely large in the most demanding scenarios. In particular, when $(n,p)=(200,100)$ and $(300,150)$ the standard deviation exceeds 24 and 28 in the Linear Baseline design and reaches values above 30 in the Linear Sparse Correlated design.

Relative to OLS, deviations for Lasso and Elastic Net are modest but systematic: mean t-statistics are slightly positive and standard deviations exceed one in most scenarios, consistent with the moderate undercoverage reported in Table 3. Overall, these deviations remain limited, with means close to zero and standard deviations near one. This suggests that regularized nuisance estimation yields more stable standard error estimates and improves the finite-sample reliability of DML inference.

::: {=latex}
\begin{figure}[!htbp]
\centering
\includegraphics[width=0.95\linewidth]{../outputs/t-stat_section/fig_tstat_mean_grouped_bar.png}
\caption{Mean t-statistics by scenario and learner with zero reference line.}
\label{fig:tstat-mean-grouped-bar}
\end{figure}
:::

Figure 6 shows the mean of the t-statistics. For Lasso and Elastic Net, the mean t-statistics remain relatively close to zero  across most scenarios and in both data-generating processes. This indicates that the standardized estimates produced by the regularized learners are reasonably well centered, suggesting that the corresponding standard errors are broadly  consistent with the variability of the estimator.

In contrast, the OLS results display large deviations from zero in several scenarios. The most pronounced errors appear in the cases $(n,p)=(200,100)$ and $(300,150)$, where the mean t-statistics become strongly negative. These large shifts indicate that the standardized statistics are systematically inaccurate in these scenarios. Although the pattern is less extreme in other cells, OLS still shows greater instability than the regularized learners.

To sum up, these results help explain the severe coverage failures observed in the previous section. When the nuisance functions are estimated using OLS, the cross-fitting procedure effectively reduces the training sample to roughly half of the available observations. In scenarios where the number of controls is close to the fold-level sample size, the OLS nuisance regressions become poorly conditioned and highly unstable. This instability passes through the DML procedure, producing extreme results and inaccurately estimated standard errors. Consequently, the resulting t-statistics differ from their theoretical distribution, leading to undercoverage of the confidence intervals. Regularized learners reduce this problem by stabilizing the nuisance estimation step, which in turn produces more reliable standard errors and better calibrated inference.

## 5. Empirical Application

### 5.1 Data description

The empirical analysis uses 401(k) eligibility dataset originally studied by Poterba, Venti, and Wise (1994). The dataset contains information on U.S. households and is frequently used to study the causal effect of retirement plan eligibility on household wealth.

The outcome variable in the analysis is **household financial wealth**, measured by the variable `net_tfa`. The treatment variable is **401(k) eligibility**, denoted by `e401`, which is a binary indicator equal to one if the household is eligible to participate in a 401(k) retirement plan and zero otherwise. Eligibility is determined by the employer and therefore provides a source of variation that can be used to study how access to retirement savings plans affects wealth accumulation.

The dataset includes **9,915 households** and **11 observed variables**. In addition to the treatment and outcome variables, the covariate vector contains several demographic and economic characteristics of households. These include age (`age`), household income (`inc`), family size (`fsize`), education (`educ`), participation in an Individual Retirement Account (`pira`), home ownership (`hown`), marital status (`marr`), defined-benefit pension coverage (`db`), and an indicator for households with two earners (`twoearn`). These variables capture key socio-economic factors that may influence both retirement plan eligibility and wealth accumulation.

::: {=latex}
\input{../outputs/empirical_section/summary_statistics.tex}
:::

Several features of the data are noteworthy. First, household wealth exhibits large dispersion across observations. The mean level of financial wealth is approximately 18,051, while the standard deviation exceeds 63,000, indicating that the distribution is heavy-tailed with significant variation across households. The observed range is also very wide, with both negative wealth values and extremely large positive values. This heavy-tailed distribution is typical in household wealth data and reflects large differences in financial positions across households.

Second, the treatment variable indicates that approximately 37 percent of households are eligible for a 401(k) plan. This share provides sufficient variation in treatment status to study the effect of retirement plan access on wealth outcomes.

Third, the demographic control variables show meaningful heterogeneity in the sample. The average age of the household reference person is around 41 years, with values ranging from 25 to 64, indicating that the sample primarily consists of working-age households. Household income also displays significant variation, which is important because income is strongly related to both savings behavior and retirement plan participation. The presence of multiple demographic and economic controls allows flexible estimation of the nuisance functions.

### 5.2 Model Specification

This section describes how the Double Machine Learning (DML) framework is applied to the 401(k) dataset to estimate the causal effect of retirement plan eligibility on household financial wealth.

The empirical analysis follows the partially linear model introduced in Section 3. In this setting, $Y_i$ denote the financial wealth of household $i$, $D_i$ denote the eligibility indicator, and $X_i$ denote the vector of observed household characteristics. The empirical model can therefore be written as

$$
Y_i = \theta_0 D_i + g_0(X_i) + U_i,
$$

where $\theta_0$ represents the causal effect of 401(k) eligibility on household wealth, $g_0(X_i)$ is an unknown function capturing the effect of household characteristics, and $U_i$ is an error term. The parameter of interest is $\theta_0$, which measures how access to a retirement savings plan influences the level of financial assets held by a household.

The treatment variable is modeled as

$$
D_i = m_0(X_i) + V_i,
$$

where $m_0(X_i) = E[D_i \mid X_i]$ represents the conditional probability that a household is eligible for a 401(k) plan given its observed characteristics. The function $m_0(\cdot)$ is unknown and must be estimated from the data.

Following the Double Machine Learning procedure, both nuisance functions $g_0(X)$ and $m_0(X)$ are estimated in a first stage using machine learning methods. These estimates are then used to construct residualized variables

$$
\tilde{Y}_i = Y_i - \hat{g}(X_i), \qquad
\tilde{D}_i = D_i - \hat{m}(X_i).
$$

The treatment effect estimator is obtained by regressing the residualized outcome on the residualized treatment:

$$
\hat{\theta} =
\frac{\frac{1}{n}\sum_{i=1}^{n} \tilde{D}_i \tilde{Y}_i}
{\frac{1}{n}\sum_{i=1}^{n} \tilde{D}_i^2}.
$$

This orthogonalization step reduces the sensitivity of the treatment effect estimator to errors in nuisance estimation.

To avoid overfitting, the empirical analysis implements **cross-fitting with two folds**, consistent with the simulation design used earlier in the paper. The sample is randomly divided into two parts. For each fold, the nuisance functions are estimated on one subsample and predictions are generated for the observations in the other subsample. The treatment effect estimator is then computed by combining the residualized variables across folds. This procedure ensures that the predictions used in the second stage are generated from models estimated on independent data.

The empirical analysis compares three alternative estimators for the nuisance functions:

- **Ordinary Least Squares (OLS)**
- **Lasso regression**
- **Elastic Net regression**

All three estimators are applied within the same Double Machine Learning procedure so that the only difference across specifications is the method used to estimate the nuisance functions. This allows us to directly evaluate how the choice of nuisance estimator affects the estimated treatment effect and the reliability of statistical inference in the empirical application.

### 5.3 Results

Table 5 reports the estimated treatment effect, the corresponding standard error, and the 95% confidence interval for each specification.

::: {=latex}
\input{../outputs/empirical_section/real_data_results.tex}
:::

Across all specifications, the estimated treatment effect is positive, indicating that eligibility for a 401(k) plan is associated with higher levels of household financial wealth. This result is consistent with the interpretation that access to retirement savings plans encourages households to accumulate more financial assets.

However, the estimates differ across nuisance estimators. The OLS specification produces results that differ from those obtained using the regularized learners and is accompanied by different standard errors. In contrast, the estimates obtained using Lasso and Elastic Net are more similar to each other and exhibit more stable inference.

These differences illustrate an important point highlighted in the simulation study. Even when the underlying model is the same, the choice of nuisance estimator can influence the stability of the final treatment effect estimator and the reliability of the associated confidence intervals.

In particular, the results are consistent with the role of fold-level dimensionality in shaping finite-sample behavior. Even in a setting with moderate overall dimensionality, regularized nuisance estimators produce more stable standard errors, whereas unregularized OLS exhibits greater sensitivity to the estimation of nuisance functions. This suggests that the reliability of DML inference in practice depends critically on how nuisance components are estimated, rather than on the choice of model alone.

### 5.4 Discussion

The empirical results provide additional insight into how the choice of nuisance estimator affects inference in applied settings. Across all specifications, the estimated effect of 401(k) eligibility on household financial wealth is positive, suggesting that access to employer-sponsored retirement plans is associated with higher levels of financial asset accumulation.

At the same time, the comparison across nuisance estimators highlights differences in the stability of the estimates. While the treatment effect estimates obtained using Lasso and Elastic Net are relatively similar, the results produced by OLS exhibit greater variability in the estimated standard errors and confidence intervals.

An explanation for this behavior is that the empirical dataset contains several correlated covariates and variables with considerable variation across households. In such settings, regularization can help stabilize the nuisance estimation stage by shrinking coefficients and reducing sensitivity to noise in the data. As a result, the final treatment effect estimator becomes less sensitive to extreme realizations in the nuisance predictions.

Although the empirical results do not display the dramatic instability observed in some simulation scenarios, they still illustrate that the choice of nuisance estimator can affect the reliability of the estimated treatment effects and the associated inference. In practice, this suggests that researchers applying Double Machine Learning methods should carefully consider the specification of the nuisance models and evaluate whether regularization may improve the stability of their results.

In general, the empirical findings support the main message of this paper: the performance of Double Machine Learning estimators in finite samples can depend on how the nuisance functions are estimated. Even in moderate-dimensional settings where the number of controls is smaller than the sample size, the use of regularized learners may provide more stable and reliable inference.

## 6. Conclusion

This paper examined how the choice of nuisance estimator affects the finite-sample performance of the Double Machine Learning (DML) estimator in partially linear models. The results show that, despite strong asymptotic guarantees, finite-sample behavior is driven by the stability of nuisance estimation under cross-fitting. In particular, we find that the relevant measure of dimensionality is the fold-level ratio $pK/n$. Even when $p < n$, cross-fitting reduces the effective sample size to $n/K$, which can lead to instability in nuisance estimation. As a result, moderate-dimensional settings can exhibit substantial distortions in inference. When $pK/n$ approaches one, the nuisance regression becomes unstable, leading to large variance in the estimated residuals and, consequently, to unreliable inference.

The main results emerge from a Monte Carlo simulation study comparing three nuisance estimation strategies: ordinary least squares (OLS), Lasso, and Elastic Net. The simulations reveal that bias is not the primary source of estimation error in this setting. Across all scenarios, the estimated treatment effects remain close to the true parameter value, with average bias typically below $5\%$ of the true effect. This indicates that the orthogonalization step in the DML procedure successfully protects the estimator from systematic bias arising from nuisance estimation errors.

However, the results show large differences in estimator dispersion. In several scenarios, particularly when the number of covariates becomes large relative to the fold-level sample size used in cross-fitting, OLS produces extremely large estimation errors. These errors arise from instability in nuisance estimation and appear as occasional extreme realizations. As a result, the distribution of the OLS estimator becomes heavy-tailed and highly dispersed, even though its median remains close to the true parameter. On the other hand, the distributions produced by Lasso and Elastic Net remain tightly concentrated around the true effect, with considerably smaller variability across simulation replications.

These differences in dispersion translate directly into differences in overall estimation accuracy. The root mean squared error results show that the regularized learners consistently outperform OLS across all scenarios. In the most unstable cases, the RMSE of the OLS estimator is almost an order of magnitude larger than that of the regularized estimators. The advantage of regularization is therefore variance control, not bias reduction.

The consequences for statistical inference are even more pronounced. Confidence interval coverage results reveal severe undercoverage when OLS is used to estimate the nuisance functions. In some scenarios the empirical coverage of nominal 95% confidence intervals falls to values around 0.15–0.19. Diagnostic analysis of the corresponding t-statistics shows that these failures arise because the estimated standard errors do not adequately capture the true variability of the estimator. When OLS nuisance regressions become unstable, the resulting treatment effect estimates occasionally take extreme values, causing the t-statistics to deviate dramatically from their theoretical distribution.

Regularized nuisance estimators significantly reduce these problems. Across all simulation scenarios, Lasso and Elastic Net produce t-statistics that are substantially closer to the theoretical benchmark, although small differences in both centering and dispersion remain. Consequently, their confidence interval coverage remains much closer to the nominal level. These findings indicate that regularization stabilizes the nuisance estimation stage and improves the reliability of inference within the DML framework.

The empirical results are consistent with the simulation findings: while point estimates are similar across specifications, regularized estimators produce more stable standard errors, indicating improved finite-sample reliability of inference.

The results imply that reliable DML inference in finite samples requires controlling the fold-level ratio $pK/n$. In settings where this ratio is non-negligible, unregularized nuisance estimation can lead to unstable behavior even when $p < n$. Regularization provides a practical way to reduce this problem.

## References

Bach, Philipp, Oliver Schacht, Victor Chernozhukov, Sven Klaassen, and Martin Spindler. 2024. "Hyperparameter Tuning for Causal Inference with Double Machine Learning: A Simulation Study." In *Proceedings of the Third Conference on Causal Learning and Reasoning*, *Proceedings of Machine Learning Research* 236: 1065--1117. PMLR.

Ballinari, Dario, and Noemi Bearth. 2024. "Improving the finite sample performance of double/debiased machine learning with propensity score calibration." arXiv:2409.04874.

Belloni, Alexandre, Victor Chernozhukov, and Christian Hansen. 2014. "Inference on Treatment Effects after Selection among High-Dimensional Controls." *The Review of Economic Studies* 81 (2): 608--650.

Buhlmann, Peter, and Sara van de Geer. 2011. *Statistics for High-Dimensional Data: Methods, Theory and Applications.* Springer.

Chernozhukov, Victor, Christian Hansen, Nathan Kallus, Martin Spindler, and Vasilis Syrgkanis. 2024. *Applied Causal Inference Powered by Machine Learning and AI.* Springer. https://doi.org/10.1007/978-3-031-64316-9

Chernozhukov, Victor, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, Whitney Newey, and James Robins. 2018. "Double/debiased Machine Learning for Treatment and Structural Parameters." *The Econometrics Journal* 21 (1): C1--C68.

Farrell, Max H. 2015. "Robust Inference on Average Treatment Effects with Possibly More Covariates than Observations." *Journal of Econometrics* 189 (1): 1--23.

Naghi, Amin. 2021. "Finite Sample Evaluation of Causal Machine Learning Methods: Guidelines for the Applied Researcher." *Tinbergen Institute Discussion Paper 2021-090*.

::: {=latex}
\clearpage
:::

## 7. Appendix

::: {=latex}
\input{../outputs/bias_section/publication/table_bias_appendix_full.tex}
:::

::: {=latex}
\input{../outputs/t-stat_section/tstat_diagnostics_table.tex}
:::

::: {=latex}
\clearpage
\begin{figure}[!htbp]
\centering
\includegraphics[width=0.95\linewidth]{../outputs/bias_section/publication/fig_bias_combined_heatmaps.png}
\caption*{Figure B1: Combined bias heatmaps by scenario and learner.}
\noindent\textit{Notes: The figure shows bias across Monte Carlo scenarios for each estimator. Bias is small in all cases. OLS remains centered around zero, while Lasso and Elastic Net exhibit positive bias due to shrinkage. These patterns confirm that differences in performance are not driven by bias, but by instability and variance effects.}
\end{figure}
\clearpage
\begin{figure}[!t]
\centering
\includegraphics[width=0.92\linewidth]{../outputs/checks/fig_theta_hat_ols_outliers.png}
\caption*{Figure B2: OLS $\hat{\theta}$ outliers in instability scenarios.}
\noindent\textit{Notes: The figure reports root mean squared error (RMSE) across Monte Carlo scenarios for each estimator. OLS exhibits substantially higher RMSE in settings with large $pK/n$, reflecting instability in nuisance estimation. In contrast, Lasso and Elastic Net achieve lower and more stable RMSE due to regularization, indicating improved finite-sample accuracy despite the presence of shrinkage bias.}
\end{figure}
\begin{figure}[!htbp]
\centering
\includegraphics[width=0.95\linewidth]{../outputs/coverage_section/figures/fig_coverage_line.png}
\caption*{Figure B3: Coverage line plot across scenarios and learners.}
\noindent\textit{Notes: The figure shows empirical coverage rates relative to the nominal 95\% level. OLS exhibits severe undercoverage in scenarios with high $pK/n$, indicating significant underestimation of variability. Regularized estimators achieve coverage closer to the nominal level, although slight  undercoverage persists, reflecting residual finite-sample distortions.}
\end{figure}
\begin{figure}[!htbp]
\centering
\includegraphics[width=0.90\linewidth]{../outputs/t-stat_section/fig_tstat_sd_heatmap.png}
\caption*{Figure B4: Heatmap of t-statistic standard deviations by scenario and learner.}
\noindent\textit{Notes: The figure displays the distribution of $t$-statistics across Monte Carlo replications for each estimator. OLS exhibits over-dispersion and heavy tails, particularly in scenarios with high $pK/n$, indicating high instability in nuisance estimation. Lasso and Elastic Net produce distributions closer to the benchmark.}
\end{figure}
\begin{figure}[!htbp]
\centering
\includegraphics[width=0.98\linewidth]{../outputs/t-stat_section/fig_tstat_distribution_n200_p100.png}
\caption*{Figure B5: t-statistic distribution for scenario $(n,p)=(200,100)$.}
\noindent\textit{Notes: The figure reports the mean of the $t$-statistics across Monte Carlo replications. Values close to zero indicate correct centering. OLS remains approximately centered, while Lasso and Elastic Net exhibit a systematic positive shift, reflecting residual finite-sample bias induced by regularization.}
\end{figure}
\begin{figure}[!htbp]
\centering
\includegraphics[width=0.98\linewidth]{../outputs/t-stat_section/fig_tstat_distribution_n300_p150.png}
\caption*{Figure B6: t-statistic distribution for scenario $(n,p)=(300,150)$.}
\noindent\textit{Notes: The figure shows the standard deviation of the $t$-statistics. Under correct inference, the standard deviation should be equal to one. OLS exhibits considerable over-dispersion, with standard deviations far exceeding one in scenarios with high $pK/n$, indicating severe inaccuracy of standard errors. Lasso and Elastic Net are closer to the benchmark but still show moderate over-dispersion.}
\end{figure}
:::
