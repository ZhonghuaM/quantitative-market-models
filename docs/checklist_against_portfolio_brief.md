# Checklist Against Portfolio Brief

## Implemented

- Package-first Python repository with tests, CI, Docker, Makefile, and examples.
- Recruiter-facing README, methodology, model cards, assumptions, architecture, and AI risk notes.
- Portfolio optimization: mean-variance, tangency, minimum variance, risk parity, hierarchical risk parity, CVaR minimization, and equal weight.
- Risk analytics: VaR, Expected Shortfall, drawdown, stress testing, Kupiec backtest, rolling ES, and volatility regimes.
- Derivatives: Black-Scholes, binomial tree, Monte Carlo option pricing, Greeks, implied volatility, convergence, and surfaces.
- Backtesting: walk-forward validation, transaction costs, no-lookahead feature construction, and model comparison.
- Machine learning: naive baseline, logistic regression, SVM, gradient boosting, random forest, calibration, and feature importance.
- Time-series tools: AR(1), exponential smoothing, Kalman local-level filtering, regime transition matrix, and pair-spread signal utilities.
- Factor models: CAPM regression, rolling beta, PCA statistical factors.
- NLP/AI: source-grounded TF-IDF retrieval demo and responsible-AI notes.
- Public data workflows: Stooq market data, SEC company facts, and FRED graph CSV downloader.
- Multi-language demonstrations: C++, JavaScript, R, and SQL.

## Deliberately Not Yet Added

- Live trading or brokerage integration.
- Heavy deep-learning frameworks such as PyTorch.
- A full SEC filings RAG repository with embeddings and LLM generation.
- A dashboard server such as Streamlit or FastAPI.

Those are better as separate repositories or optional second-stage modules so this flagship remains readable.
