# Model Cards

## Walk-forward Direction Classifier

Purpose: Demonstrate disciplined financial ML research on noisy daily market data.

Inputs: OHLCV-derived returns, ranges, volume features, volatility, moving-average ratios, EMA spread, and RSI.

Output: Probability of a positive next-day close-to-close return.

Validation: Expanding-window walk-forward validation with fixed test blocks.

Decision rule: Long when predicted probability is at least 0.53; otherwise flat.

Known limitations: The sample dataset is short, features are simple technical features, and reported performance is a historical hypothetical backtest.

## EWMA VaR Model

Purpose: Estimate and backtest horizon loss thresholds from rolling volatility.

Inputs: Daily close-to-close returns.

Output: 99% 10-day VaR threshold and breach table.

Validation: Kupiec proportion-of-failures test.

Known limitations: Normal VaR with EWMA volatility does not fully model fat tails, jumps, or changing correlations.

## Derivatives Pricer

Purpose: Demonstrate numerical consistency between closed-form, tree, and Monte Carlo methods.

Inputs: Spot, strike, rate, volatility, maturity, and simulation/tree parameters.

Output: Option value, Greeks, implied volatility, convergence tables, and surfaces.

Known limitations: Examples assume constant volatility/rates and frictionless markets.
