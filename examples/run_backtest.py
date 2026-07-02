import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_models.backtest import backtest_probability_signal, performance_metrics
from quant_models.data import load_ohlcv
from quant_models.features import build_trend_dataset
from quant_models.ml import walk_forward_classification


def main() -> None:
    data = load_ohlcv()
    dataset, features = build_trend_dataset(data)
    result = walk_forward_classification(dataset, features)
    backtest = backtest_probability_signal(result.predictions)
    print(performance_metrics(backtest["strategy_return"]))


if __name__ == "__main__":
    main()
