# Architecture

```mermaid
flowchart LR
    A["data/*.csv or optional download scripts"] --> B["data loaders"]
    B --> C["feature engineering"]
    C --> D["walk-forward ML models"]
    C --> E["risk analytics"]
    C --> F["factor and volatility diagnostics"]
    D --> G["transaction-cost backtest"]
    E --> H["risk report tables and figures"]
    F --> H
    G --> H
    I["portfolio and derivatives modules"] --> H
    H --> J["reports/summary.md and figures"]
```

The code is organized as a small research platform: reusable package modules under `src/quant_models`, executable workflows under `scripts/`, communication artifacts under `docs/`, and generated outputs under `reports/`.
