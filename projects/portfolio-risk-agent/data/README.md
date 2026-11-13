# Synthetic fixture

`synthetic_prices.csv` contains 900 date-labelled price rows for three invented assets: `SYNTH_EQUITY`, `SYNTH_DEFENSIVE`, and `SYNTH_BOND`. These are simulated series, not ticker symbols or market observations. Dates run from 2020-01-02 to 2023-06-14 and include Monday–Friday labels irrespective of real exchange holidays.

`generate_synthetic.py` uses NumPy's PCG64 generator with seed 20261113. Prices start at 100 and evolve by correlated Gaussian log increments with three deterministic, mildly different drift/volatility regimes. `manifest.json` records parameters, regime boundaries, the generator hash, the file hash, and the NumPy version. There are no downloaded inputs, corporate actions, dividends, transaction costs, or missing rows.

From the project directory:

```bash
python data/generate_synthetic.py
```

This rewrites the fixture and manifest. Committed bytes are the canonical test input; numerical-library differences can affect regenerated final digits. The fixture is MIT-licensed and demonstrates numerical and software behavior. Its simulated returns are not evidence of market performance, investment value, or real assets managed.
