# Quantitative Market Models

Reusable Python models for market analysis, risk measurement, derivative pricing, portfolio construction, and a walk-forward equity trading signal.

The repository is designed as a clean research package rather than a notebook archive. It includes sample datasets, reusable modules under `src/quant_models`, an end-to-end analysis runner, generated reports, and tests.

## What is included

- Portfolio optimization: covariance construction, global minimum-variance portfolio, tangency portfolio, and sampled efficient-frontier diagnostics.
- Risk analysis: historical return handling, EWMA volatility, 99% value-at-risk backtesting, and Kupiec breach-rate testing.
- Derivatives: Black-Scholes European call pricing, CRR binomial convergence, and Monte Carlo arithmetic Asian call pricing.
- Stochastic simulation: exact, Euler-Maruyama, and Milstein geometric Brownian motion paths.
- Numerical methods: finite-difference boundary-value solver and plain Monte Carlo integration.
- Trading model: engineered OHLCV features, expanding-window random-forest classification, probability-threshold signals, transaction costs, equity curves, and drawdown metrics.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/run_analysis.py
pytest
```

The analysis runner writes regenerated outputs to `reports/`:

- `reports/summary.md` - concise human-readable result summary.
- `reports/metrics.json` - machine-readable metrics.
- `reports/signal_backtest.csv` - out-of-sample predictions, positions, returns, and equity curves.
- `reports/feature_importance.csv` - mean feature importances across walk-forward folds.
- `reports/ewma_var_backtest.csv` - risk-model validation table.
- `reports/portfolio_summary.csv` - tangency and minimum-variance portfolio diagnostics.
- `reports/figures/` - publication-ready PNG figures.

## Repository layout

```text
.
├── data/
│   ├── sp500_index.csv
│   └── vas_equity_etf.csv
├── reports/
│   └── figures/
├── scripts/
│   └── run_analysis.py
├── src/
│   └── quant_models/
└── tests/
```

## Method notes

The trading model uses features available at date `t` to predict the close-to-close move from `t` to `t+1`. The out-of-sample evaluation uses an expanding training window with fixed test blocks, avoiding random shuffling of time-series observations.

The default signal is long/flat:

- Long when the predicted probability of an up day is at least `0.53`.
- Flat otherwise.
- Transaction cost is modeled as 5 basis points per unit of position turnover.

All results are historical and illustrative. This repository is research code, not investment advice.
