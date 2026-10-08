# Methodology

## Data and targets

The signal study uses the bundled OHLCV file. Observed schema, dates, missing values, duplicates, and SHA-256 hashes are recorded in the [data manifest](../data/manifest.json). Provider, retrieval date, instrument identity, exchange, calendar, adjustments, and data licence are unverified. Optional download scripts do not establish the bundled files' provenance.

The direction experiment uses `Close`, not `Adj Close`. For row t, the target return is `Close[t+1] / Close[t] - 1`, and the positive class is a strictly positive return. Here t+1 means the next recorded observation, not necessarily the next calendar day. Features use price, range, volume, moving-average, volatility, and RSI information available through row t. Initial rolling-window rows and rows with unavailable features or forward returns are excluded.

## Walk-forward prediction

All models use the same initial 504-row training sample and successive 63-row test blocks. Training expands at each block boundary. A model stays fixed within its test block while each prediction uses that row's available features. At the first test row t, the final training label is the t−1 to t return, known only after close t; refitting therefore occurs after that close. Standardization, where used, is fitted within each training fold.

The model set contains training-frequency and most-frequent baselines, logistic regression, linear SVM, histogram gradient boosting, and a random forest. The forest uses 300 trees, maximum depth 5, minimum leaf size 20, and seed 42. Classification uses probability 0.5 as its cutoff; the trading rule uses a separate 0.53 cutoff. Comparison and trading share stored random-forest predictions. The forest is fixed as the trading example, not selected by the comparison ranking. The training-frequency baseline predicts the positive-label fraction from its training fold; the majority baseline instead assigns probability 1 to its most frequent class.

Accuracy, balanced accuracy, ROC AUC, and Brier score measure different aspects of prediction. Class-weighted forest and logistic outputs are uncalibrated probability scores. Calibration bins include 0 and 1. SVM probability estimation relies on the estimator's internal calibration within each training fit; it is not a separate chronological calibration study. No method in this comparison has a separately untouched final holdout.

For prediction uncertainty, each row's improvement is the training-frequency baseline's squared probability error minus the forest's squared error. A paired moving-block bootstrap resamples this loss-difference series, using 1,000 draws, seed 42, and block lengths 10 and 20. The 95% percentile intervals are exploratory and conditional on fitted predictions; they do not include refitting uncertainty or correction for model selection.

## Trading clock and accounting

Let `signal[t]` be the long/flat decision computed after observing all inputs through close t, and `forward_return[t]` the close t to close t+1 return.

- **Main delayed-close illustration:** `position[t] = signal[t-1]`, with the first position flat. The previous close's signal is executed at close t and earns `forward_return[t]`.
- **Same-close diagnostic:** `position[t] = signal[t]`. This assumes observation of the final close, high, low, and volume while obtaining that same close as the fill; it is not an established executable strategy.

Both use the same prediction rows. A one-close delay is a persistence/execution sensitivity of the original forecast, not a new evaluation target or evidence of calibration for a longer horizon.

Gross return is position times forward return. A 5-basis-point charge applies to absolute position changes, including entry from cash. There is no final liquidation charge. Backtest dates label the start of each forward-return interval, not settlement at the following close. Returns compound from initial wealth 1, and drawdown includes this initial wealth in the running peak. Sharpe uses annualized arithmetic excess return divided by annualized return volatility; CAGR is reported separately. Annualization uses 252 observations per year.

## Supplementary risk example

This example runs with `--study all` and is separate from the direction experiment. `ewma_volatility` forecasts return t using only returns before t. By default, the first 20 returns initialize the population variance and receive no forecasts. An externally supplied initial variance must come from information available before the series.

`ewma_var_backtest` reports an after-close forecast: at row t it updates the prior variance with return t and compares the resulting VaR with the loss over t+1 through t+h. The report uses h=1 and 99% normal VaR, with EWMA decay 0.94. This avoids the mechanical overlap of rolling multi-day losses. Kupiec's unconditional-coverage statistic handles zero and all breaches using limiting likelihoods; its chi-square p-value remains an asymptotic diagnostic with a Bernoulli independence assumption. One-day horizons do not establish independence.

The ten-day normal VaR/Expected Shortfall calculation is a separate distributional illustration, not the horizon used for the Kupiec result. Rolling historical Expected Shortfall and stress scenarios describe additional tail-risk assumptions.

## Other worked examples and verification

Portfolio examples use a small stylized universe with assumed expected returns, volatilities, and correlations; simulated scenarios support the CVaR example. These are not estimated investable recommendations. Black–Scholes and CRR trees share assumptions for convergence checks; Asian-option Monte Carlo is a separate payoff example. GARCH fitting, PCA, and time-series diagnostics are descriptive sample analyses, not additional out-of-sample trading signals.

Tests check causal feature/volatility behavior, boundary likelihoods, target alignment, initial drawdown, fold overlap handling, calibration endpoints, labelled portfolio ordering, and numerical identities. They validate specified behavior, not the economic validity of the results.
