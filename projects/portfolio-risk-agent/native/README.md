# Native numerical engine

The calculation layer uses C17 and C++17 with the standard library only. C owns
the deterministic random generator and normal transform. C++ owns CSV validation,
sample statistics, portfolio optimisation, simulation, causal VaR backtesting and
paired uncertainty calculations. Python invokes the executable and prepares the
report; it does not substitute Python calculations for these native results.

## Build and check

From the project directory:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 2
ctest --test-dir build --output-on-failure
build/qfa_engine version

cmake -S . -B build-sanitize -DCMAKE_BUILD_TYPE=Debug -DQFA_SANITIZERS=ON
cmake --build build-sanitize -j 2
ctest --test-dir build-sanitize --output-on-failure
```

The sanitizer option requires Clang or GCC and enables AddressSanitizer and
UndefinedBehaviorSanitizer. Tests use explicit runtime checks, which remain
enabled in release builds. They cover analytic allocations, singular covariance,
boundary portfolios, fractional tail mass, forecast chronology, Kupiec boundary
likelihoods, seeded simulation, analytic lognormal moments and CSV rejection.

## CLI contract

```text
qfa_engine risk --prices PATH --weights 0.6,0.25,0.15 --confidence 0.99
qfa_engine optimise --prices PATH --weights 0.6,0.25,0.15 --confidence 0.99 --train 504 --frontier-points 15
qfa_engine monte-carlo --prices PATH --weights 0.6,0.25,0.15 --confidence 0.99 --paths 20000 --horizon 1 --seed 42
qfa_engine backtest --prices PATH --weights 0.6,0.25,0.15 --confidence 0.99 --window 250
```

`--prices` and `--weights` are required. The remaining values shown are defaults.
Each command rejects flags belonging to another command, duplicate flags, missing
values and nonfinite numeric inputs. The CSV header is `Date` (or `date`) followed
by one to eight distinct asset symbols. Prices must be finite and positive;
dates must be valid `YYYY-MM-DD` dates in strictly increasing order. Missing
prices are rejected. Calendar gaps are allowed: one row means one observation,
so the caller must supply the intended trading calendar. Weights are long-only
and sum to one within `1e-8`; they are interpreted in CSV column order.

Success writes one JSON object to stdout and exits zero. Failure writes
`{"error":{"code":"input_error","message":"..."}}` and exits 2, or uses
`numerical_error` and exits 3. Results use `schema_version: 1`, include asset
order, weights, dates and settings, and omit local file paths. `version` and
`--version` report the native version and compiler.

## Numerical definitions

Daily simple asset returns are `price[t] / price[t-1] - 1`. A portfolio is
rebalanced to the stated weights each observation; returns are their weighted
sum. Reported sample covariances and volatilities use denominator `n-1`.
Volatility annualisation uses `sqrt(252)`; annualised geometric return is
`expm1(252/n * sum(log1p(r)))`. These are observation-based conventions, not a
claim that every input interval is a calendar day. Transaction costs, borrowing,
taxes, liquidity and execution uncertainty are absent from this allocation
illustration.

Loss is the negative return. Historical VaR uses the type-7 linear empirical
quantile (NumPy's `method="linear"`, R's default `type=7`). ES integrates the
upper tail of the **discrete empirical loss distribution**. For `m=n*(1-c)`, it
sums the largest `floor(m)` losses and `m-floor(m)` times the next loss, then
divides by `m`. This treats a fractional boundary observation explicitly. It is
not the mean of all observations exceeding the interpolated VaR, nor the
integral of a linearly interpolated quantile curve. Negative VaR or ES is retained
when the relevant tail consists of gains.

Minimum variance uses the first `--train` return rows only. At least two training
and two holdout returns are required. The covariance used in the objective is
the training sample covariance plus `1e-8` on each daily diagonal. Every nonempty
active support is enumerated, with small equality-constrained systems solved by
partial-pivot elimination. Infeasible negative weights are rejected. Enumerating
all supports includes boundary vertices; the eight-asset limit keeps this
deliberately transparent approach bounded. Frontier targets range from the
regularised minimum-variance portfolio's mean to the highest asset mean.
Reported `estimated_daily_variance` includes the ridge; empirical training and
holdout risk metrics do not. Lower estimated variance does not establish lower
future variance or lower tail risk.

Variance contributions are `w[i] * (Sigma*w)[i]` using the unregularised sample
covariance for the corresponding period. They sum to portfolio variance.
Contributions can be negative when an asset hedges other holdings; fractions are
`null` when total variance is zero.

Monte Carlo fits sample means and covariances of asset log returns, adds `1e-12`
to the covariance diagonal for Cholesky factorisation, and draws correlated
Gaussian log returns. A simulated daily gross portfolio return is
`sum(w[i]*exp(log_return[i]))`; daily gross returns compound over the horizon.
This is a daily rebalanced multivariate lognormal model, with independent draws
across days. Paths are bounded to 100–1,000,000 and horizons to 1–20 observations.
The output exposes both raw and effective log covariance matrices. SplitMix64
uses fixed-width unsigned arithmetic; open-interval 52-bit uniforms feed cached
Box–Muller normals. Seeds reproduce results within the same build; tiny
cross-platform differences in mathematical library functions are possible.

Rolling VaR row `t` predicts the loss for the return ending at `t` from exactly
the `window` returns strictly before `t`. A breach is the strict inequality
`loss > VaR`. The Kupiec unconditional-coverage statistic handles zero/all
breaches using the limiting log likelihood and obtains its one-degree-of-freedom
chi-square tail from `erfc(sqrt(LR/2))`. This is an asymptotic coverage diagnostic;
it neither tests independence nor establishes that the Bernoulli assumption is
appropriate for a given return series.

The optimiser also reports paired moving-block bootstrap intervals on holdout
returns for **input minus candidate** annualised variance, VaR and ES. Candidate
weights remain fixed at their training fit. Each repetition samples contiguous
blocks with replacement and applies the same indices to both portfolios,
truncating the last block to retain the original sample length. Blocks 10 and 20
use 1,000 repetitions and seed 42; blocks longer than the holdout are explicitly
listed as skipped. Interval endpoints are type-7 2.5% and 97.5% bootstrap
quantiles. No models are refitted. These exploratory intervals are conditional
on the selected models, windows and sample, and can be unreliable under
nonstationarity or dependence beyond the block length.

## Result fields

All analytical commands contain `command`, `assets`, `weights`, `settings`,
`data` (`price_rows`, `return_rows`, `start_date`, `end_date`), `schema_version`
and `native_version`.

| Command | Additional fields |
| --- | --- |
| `risk` | `observations`, `mean_daily_return`, `volatility_daily`, `annualized_volatility`, `total_return`, `annualized_return`, `historical_var`, `historical_es`, `asset_mean_daily_returns`, `sample_covariance` |
| `optimise` | `training`, `holdout`, `ridge`, training moments, `minimum_variance`, `frontier`, `comparison`, `holdout_uncertainty`, `holdout_uncertainty_settings`, `holdout_uncertainty_skipped_block_sizes` |
| `monte-carlo` | `monte_carlo_var`, `monte_carlo_es`, `mean_simulated_return`, `simulated_return_std`, `estimated_log_mean`, `estimated_log_covariance`, `effective_log_covariance`, `covariance_ridge`, `generator` |
| `backtest` | `forecasts` (`date`, `var`, `loss`, `breach`), `kupiec`, `forecast_timing` |

Allocation objects contain `weights`, `estimated_daily_return`,
`estimated_daily_variance` and `target_daily_return`. `comparison.input` and
`comparison.minimum_variance` each contain weights, training/holdout risk metrics,
and `training_variance_contributions` / `holdout_variance_contributions` objects
with `absolute_daily_variance` and `fractional` arrays. Uncertainty rows contain
`metric`, `block_size`, `estimate`, `ci_lower`, `ci_upper`.
