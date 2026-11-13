# Execution and evidence contract

A run begins with a portfolio configuration supplied by the caller. The model can decide which analytical tools to invoke and recover from rejected calls, but it cannot silently change the submitted tickers, dates, weights, confidence or computational settings. A changed portfolio request starts a separately identified run.

`load_market_data` returns an artifact identifier. All later tools accept this identifier, not a filesystem path, URL or shell command. The dispatcher resolves the corresponding archived price file. The native executable is fixed by the application, invoked as an argument list without a shell, and bounded by a timeout. Unknown options, symbols, invalid weights and invalid numerical values are rejected by independent boundaries.

The OpenAI loop uses strict function schemas plus local validation, preserves call IDs and all returned output items, and returns structured tool failures. It has explicit call/round budgets. A text-only answer before successful report creation is not accepted as a completed analysis. API failures leave an auditable partial run instead of a fabricated report. Mock-client tests validate this software contract; only live cases can measure actual model selection of tools and parameters.

The report requires persisted risk, simulation, optimisation and coverage results for the same configured dataset. Its numbers are read from those result objects. The model's final text is retained in `agent_run.json`, labelled as model commentary, and does not feed the PDF or charts. Agreement of numerical reports therefore does not validate every sentence a model might generate.

## Files

| File | Evidence |
|---|---|
| `input_config.json` | Frozen portfolio, requested period and model settings |
| `data/*_prices.csv`, `*_returns.csv`, `*_metadata.json` | Actual inputs, transforms, retrieval settings and data hashes |
| `estimate_risk.json` | Native historical moments and tails |
| `simulate_risk.json` | Native scenario results, seed and fitted log moments |
| `optimise_portfolio.json` | Training interval, solution, frontier and later comparison |
| `backtest_var.json` | Past-only forecasts, realised losses and coverage diagnostic |
| `tool_audit.json` | Tool arguments, outcomes, ordering and elapsed time |
| `agent_run.json` | Mode, tool outputs, response IDs, model ID and token use when applicable |
| `run_manifest.json` | Code, data/output/binary hashes, dependencies and platform |
| `portfolio_report.pdf`, `.md`, CSVs and figures | Human-readable numerical evidence |

The source commit and dirty flag describe the code at execution time; source hashes identify uncommitted implementation changes. File hashes identify content, not a provider's accuracy. The LLM model identifier is recorded but stochastic tool choices and text are not guaranteed reproducible. Retrieved market prices may be revised; re-use the archived snapshot for numerical reproduction.

OpenAI API keys are not recorded. Yahoo prices and provider cookie/timezone caches remain in ignored local runs and are excluded from published artifacts. The checked-in outputs use only the repository's synthetic fixture. The project does not connect to a brokerage or execute trades.
