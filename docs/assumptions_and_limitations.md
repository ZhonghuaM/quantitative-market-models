# Assumptions and limitations

## Data and scope

The two bundled CSV files are small historical fixtures. Their original provider, retrieval date, instrument identity, exchange, calendar, timezone, licence, and adjustment policy cannot currently be verified. Filenames and column names do not establish these facts. See the [data record](../data/README.md). Absence of empty cells does not prove complete market-session coverage. The software's MIT licence does not establish data redistribution rights.

The prediction pipeline uses the OHLCV file's `Close` field. Dividend and corporate-action treatment is unknown, so the benchmark should not be interpreted as a verified total-return series. The stylized portfolio universe and simulated scenarios are separate from the market files and the primary study.

## Evaluation and uncertainty

The historical data have already been inspected. Chronological folds prevent training on later test rows but do not undo earlier choices made using the same history. There is no separately untouched final holdout. Metrics are exploratory; the top model in a comparison is not a confirmed winner.

Paired moving-block bootstrap intervals retain local dependence within sampled blocks. They are conditional on the fitted forecasts and do not include refitting uncertainty, research-selection effects, or protection against future regime changes. Two block lengths are only a limited sensitivity check. Small AUC improvements, apparent calibration, and positive cost-adjusted returns are distinct findings and do not establish a durable trading advantage.

## Execution and accounting

The main backtest delays the signal by one recorded close. The previous observation's completed inputs are then available, but an exact subsequent close fill is still assumed. The calculation does not model auction rules, latency, market impact, capacity, taxes, borrow constraints, or a full bid–ask spread process. The same-close comparator is an idealized timing diagnostic because final close and volume cannot automatically be assumed observable before execution at that close.

Costs are a fixed 5-basis-point turnover charge, with no final liquidation charge. The strategy is long/flat with zero assumed cash return; annualization uses 252 observations per year. The one-step prediction target is unchanged by execution delay, so the delayed signal is applied to a later interval than its original target. It is not a calibrated forecast for that later interval by construction.

## Supplementary model assumptions

- Normal VaR and square-root-of-time scaling do not capture all tail behavior, jumps, or changing dependence. Kupiec coverage does not test independence; asymptotic p-values can be unreliable with few breaches.
- Portfolio results depend on assumed expected returns, covariance, and constraints. Some methods permit short positions and some do not.
- Option-pricing checks use constant rates/volatility and idealized dynamics. Numerical agreement is conditional on these assumptions.
- GARCH, PCA, and time-series plots are descriptive fits unless evaluated under a separate forecasting design.
- The retrieval example uses illustrative text and lexical similarity; it is not an evaluated financial-document question-answering system.

## Reproduction boundary

The lock file, seeds, stored predictions, and run manifest make runs easier to inspect and compare. They do not guarantee bitwise equality across hardware, operating systems, numerical libraries, or alternative dependency installations. A data hash records input bytes; it does not authenticate a provider or resolve missing provenance. The primary study and the supplementary suite have different run scopes; consult the manifest before interpreting a directory of retained reports as one fresh run.
