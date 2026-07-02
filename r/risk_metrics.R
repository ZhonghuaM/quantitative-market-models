historical_var <- function(returns, confidence = 0.99) {
  -as.numeric(quantile(returns, probs = 1 - confidence, na.rm = TRUE))
}

historical_expected_shortfall <- function(returns, confidence = 0.99) {
  var <- historical_var(returns, confidence)
  losses <- -returns[returns <= -var]
  mean(losses, na.rm = TRUE)
}

max_drawdown <- function(returns) {
  equity <- cumprod(1 + returns)
  drawdown <- equity / cummax(equity) - 1
  min(drawdown, na.rm = TRUE)
}

if (sys.nframe() == 0) {
  set.seed(42)
  sample_returns <- rnorm(1000, mean = 0.0003, sd = 0.012)
  cat("Historical 99% VaR:", historical_var(sample_returns), "\n")
  cat("Historical 99% ES:", historical_expected_shortfall(sample_returns), "\n")
  cat("Max drawdown:", max_drawdown(sample_returns), "\n")
}
