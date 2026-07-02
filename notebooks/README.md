# Notebook Guide

The repository is intentionally package-first. Use notebooks for narrative exploration only after core logic exists in `src/quant_models`.

Suggested notebooks:

- `01_data_exploration.ipynb`
- `02_factor_research.ipynb`
- `03_ml_signal_research.ipynb`
- `04_portfolio_risk_dashboard.ipynb`
- `05_derivatives_pricing_demo.ipynb`

The production path is `python scripts/run_analysis.py`, which regenerates reports without relying on notebook state.
