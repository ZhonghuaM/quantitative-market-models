# Independent evaluation

Run these commands from the project directory after installing its Python dependencies and building the native engine:

```bash
python evaluation/evaluate.py \
  --engine build/qfa_engine \
  --output-dir outputs/evaluation
```

The evaluator actually invokes the binary and exits nonzero if a numerical check fails. It writes `evaluation_report.md`, `evaluation_report.json`, and `reference_results.csv` with measured values, tolerances, and results. The JSON records the binary, data, and reference-source hashes and Python/library versions. No expected pass counts or fabricated results are stored in advance.

The reference calculations use NumPy/SciPy independently of the native code:

- Historical VaR uses the type-7 quantile. ES integrates upper-tail empirical probability mass, including the fractional boundary observation; losses keep their signs.
- Sample moments use denominator n−1. SciPy SLSQP checks the long-only minimum-variance solution and every frontier point using exactly the same explicit 1e-8 ridge on daily covariance.
- Variance contributions are compared with finite differences of raw portfolio-return variance and must sum to that variance. Fractions sum to one for the nonzero-variance fixture; they are not constrained to be positive.
- Paired holdout bootstrap point estimates are checked against empirical risk differences. A constructed constant-return improvement must give a collapsed tail-risk improvement interval and zero variance improvement; this detects broken pairing. These checks do not establish statistical coverage of the intervals.
- Rolling forecasts are compared with strictly preceding windows. Editing the final price cannot alter earlier forecasts or the training-only allocation. Zero/all-breach inputs check Kupiec likelihood boundaries.
- A seeded simulation is repeated exactly, and a one-asset lognormal case checks simulated mean and variance against analytical values using six sampling-standard-error bounds. These are numerical sampling checks, not investment-performance intervals.

The moving-block intervals condition on fixed allocations and the observed holdout; they do not refit portfolio weights or account for research selection. Two block lengths do not establish robustness to nonstationarity. The fixture is entirely synthetic.

## Independent base-R check

After the Python evaluator exports actual native results, run:

```bash
Rscript evaluation/check_reference.R \
  --prices data/synthetic_prices.csv \
  --output-dir outputs/evaluation
```

Only base R is required. The checker reads the original prices, `native_risk_summary.csv`, and `native_covariance.csv`, then recalculates historical VaR, fractional-mass ES, sample moments/covariance, and compounded-return statistics. It does not use the Python reference values. Results are written to `r_reference_results.csv` and `r_evaluation_report.md`; failure returns a nonzero exit code. The Python report alone does not establish that R was run.

## Tool-call evaluation boundary

`test_cases.json` contains public, hand-authored intent and tool-contract examples. They can guide a recorded model evaluation but are not an untouched held-out benchmark. Schema-valid scripted calls do not establish LLM tool-selection accuracy. Report numerical checks, mocked orchestration tests, scripted runs, and actual model/API runs separately. If an API run is unavailable or fails, preserve that status rather than reporting an accuracy score.
