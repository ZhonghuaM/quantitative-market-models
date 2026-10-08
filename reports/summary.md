# Do daily OHLCV forecasts survive evaluation and execution constraints?

An exploratory study using the bundled OHLCV sample. Original vendor, download date, adjustments and redistribution terms remain unverified; see [data notes](../data/README.md).

## Prediction evidence

Evaluation signal dates: 2020-06-15 to 2023-05-18 (741 observations). All models use expanding training windows, 504 initial training rows and 63-row test blocks. Predictions are stored in predictions.csv.
The random forest is fixed as the trading example; it is not selected by the table ranking. Each block is fitted after its first test close, when the final training label is observable. Class-weighted model outputs are uncalibrated probability scores.

| Model | ROC AUC | Brier loss (lower is better) |
|---|---:|---:|
| hist_gradient_boosting | 0.538 | 0.2646 |
| linear_svm | 0.524 | 0.2462 |
| logistic_l2 | 0.512 | 0.2523 |
| naive_most_frequent | 0.500 | 0.4399 |
| random_forest | 0.525 | 0.2511 |
| training_frequency | 0.494 | 0.2465 |

The training-frequency baseline forecasts the positive-label fraction in each training fold. The majority baseline makes a hard class prediction; it is a weaker probability benchmark.

### Paired prediction-loss uncertainty

- 10-observation moving blocks: mean Brier improvement over training frequency -0.0046; 95% percentile interval [-0.0120, 0.0025].
- 20-observation moving blocks: mean Brier improvement over training frequency -0.0046; 95% percentile interval [-0.0120, 0.0026].

Positive improvement favours the forest. These paired block intervals are conditional on the stored predictions; they do not refit models or correct for development choices. A different block length or nonstationarity can alter their interpretation.

## From a forecast to an executed position

Features and forecasts become available after close t. The primary position is executed at close t+1 and earns the close t+1 to t+2 return. Row dates in the backtest label the start of the earned return, not its settlement. The first row is flat. Same-close execution is included only as a diagnostic. Both use 5 bp per unit of turnover, with no final liquidation. The delay tests persistence of the original next-day forecast; it does not retrain a model for the delayed holding period.

| Execution assumption | Total return | Arithmetic Sharpe | Max drawdown |
|---|---:|---:|---:|
| same_close_diagnostic | 20.36% | 0.687 | -16.14% |
| next_close | -0.43% | 0.035 | -15.19% |
| Buy and hold, same return rows, before costs | 25.00% | 0.597 | -17.95% |

Primary strategy exposure: 39.81%. Returns exclude cash interest, dividends unless already present in the unverified price data, market impact, and taxes. Sharpe uses annualised arithmetic mean excess return (252 periods/year; risk-free rate 0).

![Execution comparison](figures/trading_equity_curve.png)

## What the experiment establishes

This historical sample has already been used for development. Chronological model fitting prevents training on future labels but does not create a fresh final holdout. Prediction loss, calibration and cost-adjusted returns answer different questions; positive returns or a small AUC advantage do not establish a durable trading edge. The next credible step is a documented new dataset with a prespecified final evaluation, not choosing the best row above.

See run_manifest.json for configuration, environment and source/data/output hashes. See [methodology](../docs/methodology.md) for timing and limitations.

## Supplementary worked examples

These use separate datasets or stylised assumptions and are not validation of the signal study.
- One-day EWMA VaR: breach rate 2.20%; Kupiec asymptotic p-value 0.0002677. Coverage does not test independence.
- Black-Scholes call 10.4506; 200-step CRR 10.4406.
- Portfolio allocations, time-series filters, numerical methods and text retrieval are separate inspectable examples; full metrics are in metrics.json.
