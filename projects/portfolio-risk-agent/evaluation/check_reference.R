#!/usr/bin/env Rscript
# Independent base-R checks from raw prices and exported actual native values.
# No package installation, Python import, or precomputed pass result is used.

args <- commandArgs(trailingOnly = TRUE)
value <- function(flag) {
  index <- match(flag, args)
  if (is.na(index) || index == length(args)) stop(paste("Missing", flag))
  args[index + 1L]
}
prices_path <- value("--prices")
output_dir <- value("--output-dir")
prices <- read.csv(prices_path, check.names = FALSE, stringsAsFactors = FALSE)
native <- read.csv(file.path(output_dir, "native_risk_summary.csv"), stringsAsFactors = FALSE)
native_values <- setNames(native$value, native$metric)
assets <- names(prices)[-1L]
weights <- as.numeric(native_values[paste0("weight_", assets)])
confidence <- as.numeric(native_values["confidence"])
if (any(!is.finite(weights)) || abs(sum(weights) - 1) > 1e-10) stop("Invalid exported weights")
if (!is.finite(confidence) || confidence <= 0 || confidence >= 1) stop("Invalid confidence")

matrix_prices <- as.matrix(prices[, -1L, drop = FALSE])
storage.mode(matrix_prices) <- "double"
if (any(!is.finite(matrix_prices)) || any(matrix_prices <= 0)) stop("Invalid raw prices")
asset_returns <- matrix_prices[-1L, , drop = FALSE] / matrix_prices[-nrow(matrix_prices), , drop = FALSE] - 1
returns <- as.vector(asset_returns %*% weights)
losses <- -returns
var <- as.numeric(quantile(losses, confidence, type = 7, names = FALSE))
tail_count <- length(losses) * (1 - confidence)
whole <- floor(tail_count)
fraction <- tail_count - whole
descending <- sort(losses, decreasing = TRUE)
tail_sum <- if (whole > 0) sum(descending[seq_len(whole)]) else 0
if (fraction > 0) tail_sum <- tail_sum + fraction * descending[whole + 1L]
es <- tail_sum / tail_count
total_return <- prod(1 + returns) - 1
references <- c(
  observations = length(returns),
  mean_daily_return = mean(returns),
  volatility_daily = sd(returns),
  annualized_volatility = sd(returns) * sqrt(252),
  total_return = total_return,
  annualized_return = (1 + total_return) ^ (252 / length(returns)) - 1,
  historical_var = var,
  historical_es = es
)
actual <- as.numeric(native_values[names(references)])
errors <- abs(actual - references)
tolerance <- 1e-10 + 1e-8 * abs(references)
checks <- data.frame(
  check = names(references), native_value = actual, r_reference = as.numeric(references),
  absolute_error = as.numeric(errors), tolerance = as.numeric(tolerance),
  passed = is.finite(actual) & errors <= tolerance,
  stringsAsFactors = FALSE
)

native_covariance <- read.csv(file.path(output_dir, "native_covariance.csv"), check.names = FALSE, stringsAsFactors = FALSE)
if (!identical(native_covariance$asset, assets) || !identical(names(native_covariance)[-1L], assets)) stop("Covariance labels mismatch")
native_covariance_matrix <- as.matrix(native_covariance[, -1L, drop = FALSE])
expected_covariance <- cov(asset_returns)
covariance_error <- max(abs(native_covariance_matrix - expected_covariance))
checks <- rbind(checks, data.frame(
  check = "sample_covariance_ddof1_max_error", native_value = covariance_error,
  r_reference = 0, absolute_error = covariance_error, tolerance = 1e-12,
  passed = is.finite(covariance_error) && covariance_error <= 1e-12
))
write.csv(checks, file.path(output_dir, "r_reference_results.csv"), row.names = FALSE)
report <- c(
  "# Independent base-R reference check", "",
  sprintf("Executed with R %s: %d/%d numerical checks passed.", getRversion(), sum(checks$passed), nrow(checks)), "",
  "R reads the raw synthetic prices and actual native scalar/covariance exports. Historical VaR uses quantile(type = 7); ES integrates fractional upper-tail empirical mass; volatility/covariance use sample denominators. No Python reference results are used as expected values.", "",
  "See r_reference_results.csv for values, errors, and tolerances. This is numerical software validation, not an LLM accuracy or investment-performance evaluation.", ""
)
writeLines(report, file.path(output_dir, "r_evaluation_report.md"))
cat(sprintf("R checks: %d/%d passed\n", sum(checks$passed), nrow(checks)))
quit(status = if (all(checks$passed)) 0L else 1L)
