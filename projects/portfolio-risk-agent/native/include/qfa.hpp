#ifndef QFA_HPP
#define QFA_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace qfa {

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;
constexpr double optimisation_ridge = 1e-8;
constexpr double simulation_ridge = 1e-12;

struct Prices {
    std::vector<std::string> assets;
    std::vector<std::string> dates;
    Matrix values;
};

struct Returns {
    std::vector<std::string> dates;
    Matrix values;
};

struct Moments {
    Vector mean;
    Matrix covariance;
};

struct Metrics {
    std::size_t observations;
    double mean_daily_return;
    double volatility_daily;
    double annualized_volatility;
    double total_return;
    double annualized_return;
    double historical_var;
    double historical_es;
};

struct Allocation {
    Vector weights;
    double estimated_daily_return;
    double estimated_daily_variance;
    double target_daily_return;
};

struct Simulation {
    Vector portfolio_returns;
    Moments log_moments;
    Matrix effective_log_covariance;
};

struct Forecast {
    std::string date;
    double var;
    double loss;
    bool breach;
};

struct Kupiec {
    std::size_t observations;
    std::size_t breaches;
    double expected_breach_rate;
    double observed_breach_rate;
    double lr_statistic;
    double p_value;
};

struct BootstrapInterval {
    std::string metric;
    std::size_t block_size;
    double estimate;
    double ci_lower;
    double ci_upper;
};

struct VarianceContributions {
    Vector absolute_daily_variance;
    Vector fractional; // Empty if total variance is zero.
};

Prices read_prices(const std::string &path);
Returns simple_returns(const Prices &prices);
void validate_weights(const Vector &weights, std::size_t assets);
Moments moments(const Matrix &observations);
Vector portfolio_returns(const Matrix &returns, const Vector &weights);
Metrics metrics(const Vector &returns, double confidence);
double quantile(Vector values, double probability);
double expected_shortfall(Vector losses, double confidence);
double dot(const Vector &left, const Vector &right);
double quadratic_form(const Vector &weights, const Matrix &covariance);
VarianceContributions variance_contributions(const Vector &weights, const Matrix &covariance);
Matrix add_ridge(Matrix covariance, double ridge);
Allocation minimum_variance(const Moments &moments, double ridge = optimisation_ridge);
std::vector<Allocation> efficient_frontier(const Moments &moments, std::size_t points,
                                         double ridge = optimisation_ridge);
Simulation monte_carlo(const Prices &prices, const Vector &weights, std::size_t paths,
                       std::size_t horizon, uint64_t seed);
std::vector<Forecast> rolling_var(const Returns &returns, const Vector &weights,
                                  std::size_t window, double confidence);
Kupiec kupiec(const std::vector<Forecast> &forecasts, double expected_probability);
std::vector<BootstrapInterval> paired_block_bootstrap(
    const Vector &input_returns, const Vector &candidate_returns, double confidence,
    const std::vector<std::size_t> &block_sizes = {10, 20},
    std::size_t repetitions = 1000, uint64_t seed = 42);

} // namespace qfa
#endif
