#include "qfa.hpp"
#include "qfa_rng.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
std::size_t checks = 0;

void check(bool condition, const std::string &message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

void near(double actual, double expected, double tolerance, const std::string &message) {
    check(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
          message + ": actual=" + std::to_string(actual) + " expected=" + std::to_string(expected));
}

void rejects(const std::function<void()> &operation, const std::string &message) {
    bool rejected = false;
    try { operation(); } catch (const std::invalid_argument &) { rejected = true; }
    check(rejected, message);
}

void test_sample_statistics_and_tails() {
    const auto stats = qfa::moments({{1, 2}, {2, 4}, {3, 6}});
    near(stats.mean[0], 2.0, 1e-15, "sample mean");
    near(stats.covariance[0][0], 1.0, 1e-15, "sample variance uses n-1");
    near(stats.covariance[0][1], 2.0, 1e-15, "sample covariance");
    near(stats.covariance[1][1], 4.0, 1e-15, "second sample variance");
    near(qfa::quantile({1, 2, 3, 4}, 0.8), 3.4, 1e-14, "type-7 interpolation");
    near(qfa::quantile({4, 1, 3, 2}, 0), 1, 0, "minimum quantile");
    near(qfa::quantile({4, 1, 3, 2}, 1), 4, 0, "maximum quantile");
    near(qfa::expected_shortfall({1, 2, 3, 4}, 0.625), 11.0 / 3.0, 1e-14,
         "fractional boundary ES");
    near(qfa::expected_shortfall({1, 2, 3, 4}, 0.99), 4, 1e-14, "ES smaller than one observation");
    near(qfa::expected_shortfall({-4, -3, -2, -1}, 0.5), -1.5, 1e-14,
         "negative loss is not clipped");
    const auto risk = qfa::metrics({-0.1, 0.0, 0.1}, 0.95);
    near(risk.mean_daily_return, 0, 1e-15, "portfolio mean");
    near(risk.volatility_daily, 0.1, 1e-15, "sample volatility");
    near(risk.total_return, -0.01, 1e-14, "geometric total return");
    near(risk.historical_var, 0.09, 1e-14, "signed historical VaR");
    near(risk.historical_es, 0.1, 1e-14, "historical ES");
    rejects([] { qfa::metrics({0.1}, 0.99); }, "short samples rejected");
    rejects([] { qfa::expected_shortfall({1}, 1); }, "confidence boundary rejected");
    rejects([] { qfa::quantile({std::numeric_limits<double>::quiet_NaN()}, 0.5); }, "NaN rejected");
    rejects([] { qfa::validate_weights({0.5, 0.4}, 2); }, "incomplete budget rejected");
    rejects([] { qfa::validate_weights({-0.2, 1.2}, 2); }, "short position rejected");
}

void test_analytic_optimisation_and_boundaries() {
    const qfa::Moments diagonal{{0.001, 0.002}, {{0.04, 0}, {0, 0.01}}};
    const auto minimum = qfa::minimum_variance(diagonal);
    const double expected = (0.01 + qfa::optimisation_ridge) / (0.05 + 2 * qfa::optimisation_ridge);
    near(minimum.weights[0], expected, 1e-12, "analytic inverse-variance weight");
    near(minimum.weights[1], 1 - expected, 1e-12, "analytic second weight");
    near(minimum.estimated_daily_variance,
         qfa::quadratic_form(minimum.weights, qfa::add_ridge(diagonal.covariance, qfa::optimisation_ridge)),
         1e-15, "reported objective includes ridge");
    const auto frontier = qfa::efficient_frontier(diagonal, 7);
    check(frontier.size() == 7, "frontier length");
    for (const auto &point : frontier) {
        near(point.weights[0] + point.weights[1], 1, 1e-12, "frontier budget");
        near(qfa::dot(point.weights, diagonal.mean), point.target_daily_return, 1e-12,
             "frontier equality constraint");
        check(point.weights[0] >= 0 && point.weights[1] >= 0, "frontier long-only");
        check(point.estimated_daily_variance >= minimum.estimated_daily_variance - 1e-14,
              "frontier no lower than global minimum");
    }
    near(frontier.back().weights[1], 1, 1e-10, "maximum-return endpoint vertex");
    const qfa::Moments boundary{{0.001, 0.002}, {{0.01, 0.02}, {0.02, 0.09}}};
    near(qfa::minimum_variance(boundary).weights[0], 1, 1e-12, "negative unconstrained weight removed");
    const qfa::Moments duplicate{{0.001, 0.001}, {{0.04, 0.04}, {0.04, 0.04}}};
    const auto regularised = qfa::minimum_variance(duplicate);
    near(regularised.weights[0], 0.5, 1e-7, "singular sample covariance regularised");
    for (const auto &point : qfa::efficient_frontier(duplicate, 3))
        near(point.weights[0], 0.5, 1e-7, "equal means do not make target constraints singular");
    const qfa::Moments one{{0.003}, {{0}}};
    near(qfa::minimum_variance(one).weights[0], 1, 0, "one asset supported");
    near(qfa::minimum_variance(one).estimated_daily_variance, 1e-8, 1e-20, "zero covariance ridge");
    const auto contribution = qfa::variance_contributions({0.8, 0.2}, {{1, -1.5}, {-1.5, 4}});
    near(contribution.absolute_daily_variance[0], 0.4, 1e-14, "variance Euler first contribution");
    near(contribution.absolute_daily_variance[1], -0.08, 1e-14, "negative hedging contribution retained");
    near(contribution.absolute_daily_variance[0] + contribution.absolute_daily_variance[1],
         qfa::quadratic_form({0.8, 0.2}, {{1, -1.5}, {-1.5, 4}}), 1e-14,
         "variance contributions sum to portfolio variance");
    near(contribution.fractional[1], -0.25, 1e-14, "fractional hedging contribution");
    check(qfa::variance_contributions({1}, {{0}}).fractional.empty(), "zero-variance fractions undefined");
}

void test_forecast_causality_and_kupiec() {
    qfa::Returns returns;
    for (int i = 0; i < 20; ++i) {
        returns.dates.push_back(std::to_string(i));
        returns.values.push_back({(i % 2 ? 1 : -1) * 0.001 * (i + 1)});
    }
    auto edited = returns;
    for (std::size_t i = 10; i < edited.values.size(); ++i) edited.values[i][0] = -0.2;
    const auto original = qfa::rolling_var(returns, {1}, 4, 0.75);
    const auto changed = qfa::rolling_var(edited, {1}, 4, 0.75);
    check(original.size() == 16, "rolling warmup excluded");
    check(original.front().date == "4", "forecast aligned to realised return date");
    near(original.front().var, 0.0015, 1e-14, "forecast uses four strictly prior returns");
    for (std::size_t i = 0; i <= 6; ++i)
        near(original[i].var, changed[i].var, 0, "future observation cannot edit an earlier forecast");
    check(original[6].loss != changed[6].loss, "edited contemporaneous loss changes outcome only");
    std::vector<qfa::Forecast> breaches(1000, {"", 0, 0, false});
    const auto zero = qfa::kupiec(breaches, 0.01);
    near(zero.lr_statistic, 20.100671707002885, 1e-11, "zero breaches likelihood boundary");
    near(zero.p_value, 7.347086770069005e-6, 1e-16, "zero breaches must not yield p=1");
    for (std::size_t i = 0; i < 20; ++i) breaches[i].breach = true;
    const auto interior = qfa::kupiec(breaches, 0.01);
    near(interior.lr_statistic, 7.827239152922488, 1e-11, "interior Kupiec statistic");
    near(interior.p_value, 0.00514646498249972, 1e-14, "interior Kupiec p-value");
    for (auto &row : breaches) row.breach = true;
    const auto all = qfa::kupiec(breaches, 0.01);
    near(all.lr_statistic, 9210.340371976183, 1e-10, "all breaches likelihood boundary");
    check(all.p_value < 1e-100, "all breaches rejected");
}

void test_rng_and_lognormal_simulation() {
    qfa_rng generator;
    qfa_rng_seed(&generator, 0);
    check(qfa_rng_u64(&generator) == UINT64_C(0xe220a8397b1dcdaf), "SplitMix64 published seed-zero vector");
    qfa_rng_seed(&generator, 42);
    for (int i = 0; i < 1000; ++i) {
        const double value = qfa_rng_uniform(&generator);
        check(value > 0 && value < 1, "uniform excludes endpoints");
    }
    qfa::Prices prices;
    prices.assets = {"ONE"};
    prices.values = {{100}};
    for (int i = 0; i < 100; ++i)
        prices.values.push_back({prices.values.back()[0] * std::exp(i % 2 ? 0.021 : -0.019)});
    const auto simulation = qfa::monte_carlo(prices, {1}, 60000, 1, 42);
    const auto repeated = qfa::monte_carlo(prices, {1}, 60000, 1, 42);
    check(simulation.portfolio_returns == repeated.portfolio_returns, "seeded simulation repeatability");
    const double mu = simulation.log_moments.mean[0];
    const double variance = simulation.effective_log_covariance[0][0];
    const double expected_mean = std::expm1(mu + variance / 2);
    const double expected_variance = std::expm1(variance) * std::exp(2 * mu + variance);
    const auto observed = qfa::metrics(simulation.portfolio_returns, 0.99);
    near(observed.mean_daily_return, expected_mean, 6 * std::sqrt(expected_variance / 60000),
         "one-asset lognormal analytic mean within six Monte Carlo standard errors");
    near(observed.volatility_daily * observed.volatility_daily, expected_variance,
         expected_variance * 0.04, "one-asset lognormal analytic variance");
    const auto multi_day = qfa::monte_carlo(prices, {1}, 60000, 3, 17);
    const auto multi_risk = qfa::metrics(multi_day.portfolio_returns, 0.99);
    const double multi_mean = std::expm1(3 * mu + 3 * variance / 2);
    const double multi_var = std::expm1(3 * variance) * std::exp(6 * mu + 3 * variance);
    near(multi_risk.mean_daily_return, multi_mean, 6 * std::sqrt(multi_var / 60000),
         "multi-period compounding analytic mean");
}

void test_csv_validation() {
    const auto unique = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const auto file = std::filesystem::temp_directory_path() / ("qfa-test-" + std::to_string(unique) + ".csv");
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { std::error_code error; std::filesystem::remove(path, error); }
    } cleanup{file};
    auto write = [&](const std::string &text) {
        std::ofstream stream(file);
        stream << text;
        stream.close();
    };
    write("Date,\"ASSET_A\",ASSET_B\n2024-02-28,100,200\n2024-02-29,101,198\n2024-03-01,102,199\n");
    const auto prices = qfa::read_prices(file.string());
    check(prices.assets[0] == "ASSET_A" && prices.assets[1] == "ASSET_B", "asset order retained");
    near(qfa::simple_returns(prices).values[0][0], 0.01, 1e-14, "CSV close return alignment");
    const std::vector<std::string> bad = {
        "Date,A,A\n2024-01-01,1,1\n2024-01-02,1,1\n2024-01-03,1,1\n",
        "Date,A\n2024-01-01,1\n2024-01-01,2\n2024-01-03,3\n",
        "Date,A\n2024-01-02,1\n2024-01-01,2\n2024-01-03,3\n",
        "Date,A\n2023-02-28,1\n2023-02-29,2\n2023-03-01,3\n",
        "Date,A\n2024-01-01,1\n2024-01-02,nan\n2024-01-03,3\n",
        "Date,A\n2024-01-01,1\n2024-01-02,0\n2024-01-03,3\n",
        "Date,A\n2024-01-01,1\n2024-01-02,\n2024-01-03,3\n",
        "Date,A\n2024-01-01,1\n2024-01-02,2,3\n2024-01-03,3\n",
    };
    for (const auto &text : bad) {
        write(text);
        rejects([&] { qfa::read_prices(file.string()); }, "malformed CSV rejected");
    }
}

void test_paired_bootstrap() {
    const qfa::Vector input = {-0.02, 0.01, -0.01, 0.03, 0.02, -0.03};
    for (const auto &interval : qfa::paired_block_bootstrap(input, input, 0.9, {1, 3, 6}, 100, 42)) {
        near(interval.estimate, 0, 0, "paired identical portfolio point estimate");
        near(interval.ci_lower, 0, 0, "paired identical portfolio interval lower");
        near(interval.ci_upper, 0, 0, "paired identical portfolio interval upper");
    }
    qfa::Vector shifted;
    for (double value : input) shifted.push_back(value + 0.01);
    const auto constant_shift = qfa::paired_block_bootstrap(input, shifted, 0.9, {2}, 100, 42);
    near(constant_shift[0].estimate, 0, 1e-14, "constant return shift leaves variance unchanged");
    for (std::size_t i = 1; i < constant_shift.size(); ++i) {
        near(constant_shift[i].estimate, 0.01, 1e-14, "constant return improvement tail point estimate");
        near(constant_shift[i].ci_lower, 0.01, 1e-14, "paired constant improvement lower");
        near(constant_shift[i].ci_upper, 0.01, 1e-14, "paired constant improvement upper");
    }
    const qfa::Vector candidate = {-0.01, 0.005, -0.005, 0.015, 0.01, -0.015};
    for (const auto &interval : qfa::paired_block_bootstrap(input, candidate, 0.9, {6}, 100, 42)) {
        near(interval.ci_lower, interval.estimate, 0, "full-length block preserves observed estimate lower");
        near(interval.ci_upper, interval.estimate, 0, "full-length block preserves observed estimate upper");
    }
    rejects([&] { qfa::paired_block_bootstrap(input, {0.01}, 0.9); }, "paired lengths must match");
    rejects([&] { qfa::paired_block_bootstrap(input, input, 0.9, {7}); }, "oversized blocks rejected");
}
} // namespace

int main() {
    try {
        test_sample_statistics_and_tails();
        test_analytic_optimisation_and_boundaries();
        test_forecast_causality_and_kupiec();
        test_rng_and_lognormal_simulation();
        test_csv_validation();
        test_paired_bootstrap();
        std::cout << "Passed " << checks << " numerical and validation assertions.\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Native test failure: " << error.what() << '\n';
        return 1;
    }
}
