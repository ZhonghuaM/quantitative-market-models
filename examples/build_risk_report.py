from quant_models.data import close_to_close_returns, load_price_series
from quant_models.risk import ewma_var_backtest, kupiec_pof_test, parametric_var_es


def main() -> None:
    returns = close_to_close_returns(load_price_series())
    var_frame = ewma_var_backtest(returns)
    print("Kupiec:", kupiec_pof_test(var_frame["breach"], expected_probability=0.01))
    print("Parametric VaR/ES:", parametric_var_es(returns, horizon=10))


if __name__ == "__main__":
    main()
