# Quantitative Model Analysis Summary

This run regenerates the repository reports from the bundled sample data.

## Walk-forward trading model

- Out-of-sample period: 2020-06-15 to 2023-05-18
- Classification accuracy: 52.63%
- ROC AUC: 0.525
- Best baseline comparison model: hist_gradient_boosting (ROC AUC 0.530)
- Strategy total return: 20.86%
- Strategy annual volatility: 9.89%
- Strategy Sharpe ratio: 0.673
- Strategy max drawdown: -16.14%
- Buy-and-hold total return over same rows: 25.00%
- Average market exposure: 40.22%

## Risk model

- EWMA VaR observed breach rate: 1.37%
- Kupiec POF p-value: 0.213
- 10-day parametric Expected Shortfall: 5.80%

## Option and simulation checks

- Black-Scholes call value: 10.4506
- 200-step CRR value: 10.4406
- Asian call Monte Carlo value: 6.1988
- ATM delta / gamma / vega: 0.637 / 0.0188 / 0.375

## Volatility and factor diagnostics

- GARCH persistence alpha + beta: 0.947
- PCA variance explained by PC1/PC2: 0.407 / 0.213

## Time-series and retrieval diagnostics

- AR(1) phi on daily returns: -0.132
- Retrieval demo top TF-IDF score: 0.135

Research code only. Results are historical and illustrative, not investment advice.
