"""Factor-model and statistical attribution utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


def align_return_series(asset: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    """Align two return series on common dates."""

    frame = pd.concat(
        [asset.rename("asset_return"), benchmark.rename("benchmark_return")],
        axis=1,
    ).dropna()
    if frame.empty:
        raise ValueError("asset and benchmark returns have no overlapping dates")
    return frame


def capm_regression(asset: pd.Series, benchmark: pd.Series, risk_free_daily: float = 0.0) -> dict[str, float]:
    """Estimate alpha, beta, and R-squared against a benchmark return series."""

    frame = align_return_series(asset, benchmark)
    y = frame["asset_return"].to_numpy() - risk_free_daily
    x = (frame["benchmark_return"].to_numpy() - risk_free_daily).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    fitted = model.predict(x)
    residual = y - fitted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "alpha_daily": float(model.intercept_),
        "alpha_annual": float(model.intercept_ * 252.0),
        "beta": float(model.coef_[0]),
        "r_squared": float(r_squared),
        "residual_volatility_annual": float(residual.std(ddof=1) * np.sqrt(252.0)),
        "observations": float(len(frame)),
    }


def rolling_beta(asset: pd.Series, benchmark: pd.Series, window: int = 63) -> pd.Series:
    """Compute rolling CAPM beta with covariance/variance."""

    frame = align_return_series(asset, benchmark)
    covariance = frame["asset_return"].rolling(window).cov(frame["benchmark_return"])
    variance = frame["benchmark_return"].rolling(window).var()
    beta = covariance / variance
    beta.name = f"rolling_beta_{window}"
    return beta.dropna()


def pca_statistical_factors(returns: pd.DataFrame, n_components: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract PCA statistical factors from a panel of returns."""

    clean = returns.dropna(how="any")
    if clean.shape[1] < n_components:
        raise ValueError("n_components cannot exceed the number of return series")
    standardized = (clean - clean.mean()) / clean.std(ddof=0)
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(standardized)
    factor_scores = pd.DataFrame(
        scores,
        index=clean.index,
        columns=[f"pc_{i + 1}" for i in range(n_components)],
    )
    loadings = pd.DataFrame(
        pca.components_.T,
        index=clean.columns,
        columns=[f"pc_{i + 1}" for i in range(n_components)],
    )
    loadings.loc["explained_variance_ratio"] = pca.explained_variance_ratio_
    return factor_scores, loadings
