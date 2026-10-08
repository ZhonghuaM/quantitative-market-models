# Model cards

## Daily direction classifiers: primary study

**Purpose:** compare simple statistical and machine-learning methods on a shared historical direction-prediction task.

**Inputs and target:** 18 OHLCV-derived features available through each recorded close; the target is a positive return to the next recorded close. The pipeline uses `Close`; adjustment and dividend treatment of the source files are unknown.

**Evaluation:** expanding training windows, initially 504 rows, followed by 63-row test blocks. Scaling is fitted within training folds. Training-frequency and most-frequent baselines, logistic regression, linear SVM, histogram gradient boosting, and a 300-tree forest share evaluation rows. Stored predictions feed comparison metrics and the forest's trading illustration.

**Outputs:** positive-class probabilities, accuracy, balanced accuracy, ROC AUC, Brier score, calibration bins, and paired block-bootstrap Brier-loss comparisons against the training-frequency baseline. That baseline uses the training-fold positive-label fraction; the separate majority baseline emits a hard class probability. Averaged impurity-based feature importance is descriptive and can mislead when features are correlated.

**Decision rule:** long at probability ≥ 0.53 and otherwise flat, with execution delayed by one close and 5 basis points of cost per unit of turnover. A same-close calculation is a timing diagnostic. The delay changes the holding interval without changing the original prediction target.

**Limits:** small, previously inspected sample; uncertain source provenance; no untouched final holdout; no independent threshold selection; no comprehensive execution model. Class-weighted forest and logistic outputs are uncalibrated scores; probability calibration is assessed, not guaranteed. Positive returns or AUC above 0.5 are insufficient evidence of durable trading value.

## EWMA normal VaR: supplementary example

**Purpose:** illustrate an explicit forecast clock and unconditional-coverage diagnostic for tail-loss thresholds.

**Inputs:** close-to-close returns; the first 20 observations initialize variance. Subsequent volatility forecasts do not use future returns. The backtest's row-t forecast uses returns through t and targets the following observation's loss.

**Outputs:** 99% one-day normal VaR, breach indicators, breach rate, and the Kupiec statistic and asymptotic p-value. Ten-day parametric Expected Shortfall is a separate descriptive calculation.

**Limits:** normal tails, zero conditional drift in the EWMA VaR calculation, and fixed decay. Coverage alone tests neither independence nor tail-loss magnitude. Dependent breaches and sparse exceedances limit p-value interpretation. Overlapping multi-day losses should not be assessed as independent Bernoulli trials.

## Portfolio allocations: supplementary example

**Purpose:** compare algorithms under transparent, stylized inputs.

**Inputs and outputs:** assumed expected returns and covariance for four assets; simulated scenarios for CVaR; weights, expected return/volatility, and risk contributions.

**Limits:** tangency and minimum-variance solutions allow short weights; other methods impose their own constraints. Inputs are assumptions, not return forecasts validated against market data. Expected-return comparisons do not establish out-of-sample superiority.

## Derivatives and simulation: supplementary example

**Purpose:** check consistency between analytical and numerical methods.

**Inputs and outputs:** spot, strike, rate, volatility, maturity, and numerical resolution; values, Greeks, implied volatility, convergence tables, and simulation estimates.

**Limits:** constant rates/volatility, idealized price dynamics, and frictionless pricing assumptions. Asian Monte Carlo has a different payoff from the European call used for tree convergence. Numerical agreement validates the implementation under these assumptions, not observed market prices.
