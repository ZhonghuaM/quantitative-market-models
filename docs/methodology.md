# Methodology

## Data

The default analysis uses bundled sample market datasets so the project runs without credentials or network access. Optional refresh scripts are available in `scripts/download_data.py` for Stooq daily market data and SEC company-facts JSON.

## Trading Signal

The trading signal predicts next-day direction using only features available at the close of the current day. The evaluation uses expanding-window walk-forward validation. The default backtest is long/flat and includes transaction costs of 5 basis points per unit of turnover.

Models are compared with a baseline-first approach: naive classifier, logistic regression, linear SVM, histogram gradient boosting, and random forest. The report includes accuracy, balanced accuracy, ROC AUC, Brier score, calibration, and a confusion matrix.

## Risk

Risk analytics include EWMA volatility, 99% value-at-risk backtesting, Kupiec proportion-of-failures testing, rolling Expected Shortfall, and stress scenarios. VaR and ES are reported as positive loss numbers.

## Portfolio Construction

Portfolio examples include tangency, global minimum variance, equal weight, risk parity, hierarchical risk parity, and CVaR-minimization allocations. The stylized asset universe is intentionally small so the formulas are inspectable.

## Derivatives

Derivative modules include Black-Scholes pricing, CRR binomial convergence, Monte Carlo Asian option pricing, implied volatility inversion, and Greek surfaces.

## Validation

Tests cover no-lookahead feature construction, option convergence, portfolio constraints, backtest metrics, and risk diagnostics. GitHub Actions runs tests and regenerates reports on every push.
