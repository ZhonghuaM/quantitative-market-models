-- Example analytics queries for a table shaped like reports/signal_backtest.csv.
-- These are included to show how the research outputs can be queried in a
-- relational workflow after loading the CSV into a table called signal_backtest.

-- Monthly strategy and benchmark returns.
SELECT
    strftime('%Y-%m', date) AS month,
    EXP(SUM(LOG(1.0 + strategy_return))) - 1.0 AS strategy_return,
    EXP(SUM(LOG(1.0 + benchmark_return))) - 1.0 AS benchmark_return,
    AVG(position) AS average_position
FROM signal_backtest
GROUP BY 1
ORDER BY 1;

-- Largest losing days when the strategy had market exposure.
SELECT
    date,
    probability_up,
    position,
    forward_return,
    strategy_return
FROM signal_backtest
WHERE position <> 0
ORDER BY strategy_return ASC
LIMIT 10;

-- Calibration by rounded probability bucket.
SELECT
    ROUND(probability_up, 1) AS probability_bucket,
    AVG(actual_up) AS observed_up_frequency,
    COUNT(*) AS observations
FROM signal_backtest
GROUP BY 1
ORDER BY 1;
