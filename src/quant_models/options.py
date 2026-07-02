"""Option-pricing utilities."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def black_scholes_call(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
) -> float:
    """Black-Scholes value of a European call option."""

    if volatility <= 0.0 or maturity <= 0.0:
        return max(spot - strike * math.exp(-rate * maturity), 0.0)
    d1 = (math.log(spot / strike) + (rate + 0.5 * volatility**2) * maturity) / (
        volatility * math.sqrt(maturity)
    )
    d2 = d1 - volatility * math.sqrt(maturity)
    return float(spot * norm.cdf(d1) - strike * math.exp(-rate * maturity) * norm.cdf(d2))


def crr_binomial_call(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    steps: int,
) -> float:
    """Cox-Ross-Rubinstein European call value."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    dt = maturity / steps
    up = math.exp(volatility * math.sqrt(dt))
    down = 1.0 / up
    growth = math.exp(rate * dt)
    probability = (growth - down) / (up - down)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("risk-neutral probability is outside [0, 1]")

    terminal = np.array([spot * up ** (steps - i) * down**i for i in range(steps + 1)])
    values = np.maximum(terminal - strike, 0.0)
    discount = math.exp(-rate * dt)
    for _ in range(steps, 0, -1):
        values = discount * (probability * values[:-1] + (1.0 - probability) * values[1:])
    return float(values[0])


def asian_arithmetic_call_mc(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    steps: int = 12,
    paths: int = 20_000,
    seed: int = 42,
) -> dict[str, float]:
    """Monte Carlo value for an arithmetic-average Asian call."""

    rng = np.random.default_rng(seed)
    dt = maturity / steps
    shocks = rng.normal(size=(paths, steps))
    increments = (rate - 0.5 * volatility**2) * dt + volatility * math.sqrt(dt) * shocks
    log_paths = np.cumsum(increments, axis=1)
    prices = spot * np.exp(log_paths)
    averages = prices.mean(axis=1)
    discounted_payoff = math.exp(-rate * maturity) * np.maximum(averages - strike, 0.0)
    mean = float(discounted_payoff.mean())
    stderr = float(discounted_payoff.std(ddof=1) / math.sqrt(paths))
    return {"price": mean, "standard_error": stderr, "paths": float(paths)}
