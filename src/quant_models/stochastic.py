"""Stochastic-process simulation utilities."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def simulate_gbm_exact(
    spot: float,
    drift: float,
    volatility: float,
    maturity: float,
    steps: int,
    paths: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate exact geometric Brownian motion paths."""

    rng = np.random.default_rng(seed)
    dt = maturity / steps
    shocks = rng.normal(size=(steps, paths))
    increments = (drift - 0.5 * volatility**2) * dt + volatility * math.sqrt(dt) * shocks
    log_paths = np.vstack([np.zeros(paths), np.cumsum(increments, axis=0)])
    values = spot * np.exp(log_paths)
    index = np.linspace(0.0, maturity, steps + 1)
    return pd.DataFrame(values, index=index)


def simulate_gbm_euler_milstein(
    spot: float,
    drift: float,
    volatility: float,
    maturity: float,
    steps: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare Euler-Maruyama and Milstein approximations for one GBM path."""

    rng = np.random.default_rng(seed)
    dt = maturity / steps
    z = rng.normal(size=steps)
    exact = np.empty(steps + 1)
    euler = np.empty(steps + 1)
    milstein = np.empty(steps + 1)
    exact[0] = euler[0] = milstein[0] = spot

    for i in range(steps):
        dw = math.sqrt(dt) * z[i]
        exact[i + 1] = exact[i] * math.exp((drift - 0.5 * volatility**2) * dt + volatility * dw)
        euler[i + 1] = euler[i] + drift * euler[i] * dt + volatility * euler[i] * dw
        milstein[i + 1] = (
            milstein[i]
            + drift * milstein[i] * dt
            + volatility * milstein[i] * dw
            + 0.5 * volatility**2 * milstein[i] * (dw**2 - dt)
        )

    return pd.DataFrame(
        {"exact": exact, "euler": euler, "milstein": milstein},
        index=np.linspace(0.0, maturity, steps + 1),
    )
