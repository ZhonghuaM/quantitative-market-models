import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_models.data import load_ohlcv
from quant_models.time_series import ar1_fit, exponential_smoothing, kalman_local_level


def main() -> None:
    data = load_ohlcv()
    returns = data["close"].pct_change().dropna()
    print("AR(1):", ar1_fit(returns))
    print("Exponential smoothing tail:")
    print(exponential_smoothing(returns).tail())
    print("Kalman local-level tail:")
    print(kalman_local_level(returns).tail())


if __name__ == "__main__":
    main()
