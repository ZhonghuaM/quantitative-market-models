# Numerical conventions

Prices are positive finite observations on unique ascending ISO date labels. A simple asset return is `price[t]/price[t-1]-1`; portfolio return is the weighted sum, assuming fixed weights rebalanced each observation. No currency conversion or missing-session interpolation is performed. Calendar completeness is not inferred from the absence of missing cells.

## Historical tails

Loss is minus portfolio return. VaR is the linearly interpolated sample quantile (R type 7 / NumPy `method="linear"`). Signs are retained: a sample of gains may have a negative loss quantile.

For ES, give every observation probability `1/n` and integrate the empirical inverse CDF from confidence `c` to 1, dividing by `1-c`. Equivalently, average the largest `n*(1-c)` losses, with fractional weight on the boundary observation. This avoids changing tail probability merely because the sample has ties. The interpolation convention for displayed VaR differs from this empirical-distribution ES construction; it is not simply the average of every observation exceeding the displayed VaR. [Acerbi and Tasche](https://arxiv.org/abs/cond-mat/0104295) discuss why definition details matter for discrete distributions.

## Mean-variance allocation

Compute sample mean and unbiased sample covariance using the training returns only. Add `1e-8` to each diagonal element of the daily covariance. Minimise `w' Sigma w` under `sum(w)=1` and `w>=0`. Enumerate nonempty active supports for at most eight assets; solve equality-constrained systems on each face and retain the feasible minimum. Frontier points add a target-mean equality, from the global-minimum-variance mean to the maximum asset mean. Degenerate target cases and vertices need explicit treatment.

The reported model objective includes ridge; empirical risk statistics on realised returns do not. Annual arithmetic mean multiplies daily mean by 252; annual volatility multiplies sample daily standard deviation by `sqrt(252)`. CAGR and arithmetic mean are different quantities. Holdout comparisons use weights chosen from the earlier training interval and omit rebalancing costs. These are implementations of the [mean-variance framework](https://doi.org/10.1111/j.1540-6261.1952.tb01525.x), not a forecast guarantee.

## Simulation

Fit multivariate log-return sample moments. Add daily covariance ridge `1e-12` before Cholesky factorisation. The C random-number generator and Box-Muller normal transform feed correlated log-return scenarios. Each day's portfolio gross return is the weighted sum of exponentiated asset log returns; these daily portfolio gross returns compound across the requested horizon. The portfolio is rebalanced to fixed weights each day.

Seed, paths, horizon and covariance assumptions are recorded. Seeded numerical outputs are repeatable in the same environment; transcendental functions and compiler/platform differences can change final bits. The model assumes Gaussian log-return tails and stationary dependence. More simulation paths reduce Monte Carlo error, not model error.

## Coverage clock

For return observation t, the VaR forecast uses exactly the preceding `window` returns, excluding t. The realised loss at t is compared with that forecast. Tests perturb future prices to verify past forecasts cannot change. This is one-observation coverage; there is no mechanical overlap of multi-day target windows.

Kupiec's likelihood ratio uses limiting log-likelihoods at zero or all breaches. Its chi-square-one survival probability is evaluated through `erfc(sqrt(LR/2))`. The test concerns unconditional breach frequency; independence and finite-sample adequacy are separate questions. The rolling coverage test uses the original fixed portfolio, while the allocation's later tail statistics are descriptive holdout comparisons.

## Conditional uncertainty and risk contributions

The native optimiser also compares the input and fitted candidate on the same holdout dates. A paired moving-block bootstrap draws contiguous blocks of 10 and 20 observations with replacement, truncates to the original length, and recomputes annualised sample variance, VaR and ES differences. Differences are original minus candidate; positive values favour the candidate. It uses 1,000 draws and seed 42, with 95% percentile intervals. Blocks longer than the holdout are explicitly skipped. Weights remain fixed: the interval conditions on the fitted allocation and omits refitting and model-selection uncertainty. Temporal dependence beyond the chosen block and nonstationarity can invalidate nominal coverage.

Asset i's contribution to daily variance is `w[i] * (Sigma*w)[i]`, using the unregularised realised covariance. Contributions sum to portfolio variance. Fractional contributions divide by that sum; they are null for zero portfolio variance. Individual contributions may be negative when an asset hedges other holdings, even under long-only weights. Tests compare these values with independent finite differences of portfolio variance.
