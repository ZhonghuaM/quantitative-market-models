"""Reusable quantitative market-model components."""

from quant_models.backtest import performance_metrics
from quant_models.data import load_ohlcv, load_price_series
from quant_models.features import build_trend_dataset
from quant_models.factor_models import capm_regression, rolling_beta
from quant_models.ml import walk_forward_classification
from quant_models.options import black_scholes_call, black_scholes_greeks
from quant_models.portfolio import risk_parity_portfolio, tangency_portfolio

__all__ = [
    "black_scholes_call",
    "black_scholes_greeks",
    "build_trend_dataset",
    "capm_regression",
    "load_ohlcv",
    "load_price_series",
    "performance_metrics",
    "risk_parity_portfolio",
    "rolling_beta",
    "tangency_portfolio",
    "walk_forward_classification",
]

__version__ = "0.2.0"
