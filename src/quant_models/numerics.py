"""General numerical methods used by the examples."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def finite_difference_bvp(
    a: float,
    b: float,
    alpha: float,
    beta: float,
    n: int,
    p: Callable[[np.ndarray], np.ndarray | float],
    q: Callable[[np.ndarray], np.ndarray | float],
    f: Callable[[np.ndarray], np.ndarray | float],
) -> pd.DataFrame:
    """Solve y'' + p(x)y' + q(x)y = f(x) with Dirichlet boundaries."""

    if n < 2:
        raise ValueError("n must be at least 2")
    x = np.linspace(a, b, n + 1)
    dx = (b - a) / n
    matrix = np.zeros((n + 1, n + 1))
    rhs = np.zeros(n + 1)
    matrix[0, 0] = 1.0
    matrix[n, n] = 1.0
    rhs[0] = alpha
    rhs[n] = beta

    for i in range(1, n):
        xi = x[i]
        matrix[i, i - 1] = 1.0 - dx * float(p(xi)) / 2.0
        matrix[i, i] = -2.0 + dx**2 * float(q(xi))
        matrix[i, i + 1] = 1.0 + dx * float(p(xi)) / 2.0
        rhs[i] = dx**2 * float(f(xi))

    y = np.linalg.solve(matrix, rhs)
    return pd.DataFrame({"x": x, "y": y})


def monte_carlo_integral(
    function: Callable[[np.ndarray], np.ndarray],
    lower: float,
    upper: float,
    samples: int,
    seed: int = 42,
) -> dict[str, float]:
    """Estimate a one-dimensional definite integral by plain Monte Carlo."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(seed)
    x = rng.uniform(lower, upper, samples)
    values = function(x)
    width = upper - lower
    estimate = float(width * values.mean())
    stderr = float(width * values.std(ddof=1) / np.sqrt(samples))
    return {"estimate": estimate, "standard_error": stderr, "samples": float(samples)}
