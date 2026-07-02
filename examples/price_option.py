import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_models.options import (
    asian_arithmetic_call_mc,
    black_scholes_call,
    black_scholes_greeks,
    crr_binomial_call,
)


def main() -> None:
    spot, strike, rate, vol, maturity = 100.0, 100.0, 0.05, 0.20, 1.0
    print("Black-Scholes:", black_scholes_call(spot, strike, rate, vol, maturity))
    print("CRR 500-step:", crr_binomial_call(spot, strike, rate, vol, maturity, 500))
    print("Greeks:", black_scholes_greeks(spot, strike, rate, vol, maturity))
    print("Asian MC:", asian_arithmetic_call_mc(spot, strike, rate, vol, maturity))


if __name__ == "__main__":
    main()
