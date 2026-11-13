#include "qfa.hpp"
#include "qfa_rng.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>

namespace qfa {
namespace {

void require(bool condition, const std::string &message) {
    if (!condition) throw std::invalid_argument(message);
}

void confidence_valid(double confidence) {
    require(std::isfinite(confidence) && confidence > 0.0 && confidence < 1.0,
            "confidence must be strictly between zero and one");
}

std::string trim(const std::string &value) {
    const auto start = value.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    return value.substr(start, value.find_last_not_of(" \t\r\n") - start + 1);
}

std::vector<std::string> csv_row(const std::string &line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    bool closed_quote = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (quoted) {
            if (c == '"' && i + 1 < line.size() && line[i + 1] == '"') {
                field += '"';
                ++i;
            } else if (c == '"') {
                quoted = false;
                closed_quote = true;
            } else {
                field += c;
            }
        } else if (c == ',') {
            fields.push_back(trim(field));
            field.clear();
            closed_quote = false;
        } else if (c == '"') {
            require(trim(field).empty() && !closed_quote, "invalid CSV quoting");
            field.clear();
            quoted = true;
        } else {
            require(!closed_quote || c == ' ' || c == '\t' || c == '\r',
                    "characters after a closed CSV quote");
            field += c;
        }
    }
    require(!quoted, "unterminated CSV quote; multiline fields are unsupported");
    fields.push_back(trim(field));
    return fields;
}

bool valid_date(const std::string &date) {
    if (date.size() != 10 || date[4] != '-' || date[7] != '-') return false;
    for (std::size_t i = 0; i < date.size(); ++i) {
        if (i != 4 && i != 7 && (date[i] < '0' || date[i] > '9')) return false;
    }
    const int year = std::stoi(date.substr(0, 4));
    const int month = std::stoi(date.substr(5, 2));
    const int day = std::stoi(date.substr(8, 2));
    if (year < 1 || month < 1 || month > 12) return false;
    const int lengths[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    int last = lengths[month - 1];
    if (month == 2 && year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)) ++last;
    return day >= 1 && day <= last;
}

double parse_price(const std::string &text) {
    require(!text.empty(), "missing CSV price");
    errno = 0;
    char *end = nullptr;
    const double price = std::strtod(text.c_str(), &end);
    require(end == text.c_str() + text.size() && errno != ERANGE &&
            std::isfinite(price) && price > 0.0, "prices must be finite positive numbers");
    return price;
}

Vector solve(Matrix matrix, Vector rhs) {
    const std::size_t n = rhs.size();
    double scale = 0.0;
    for (const auto &row : matrix)
        for (double value : row) scale = std::max(scale, std::abs(value));
    require(scale > 0.0, "singular linear system");
    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot = col;
        for (std::size_t row = col + 1; row < n; ++row)
            if (std::abs(matrix[row][col]) > std::abs(matrix[pivot][col])) pivot = row;
        require(std::abs(matrix[pivot][col]) > 1e-13 * scale, "singular linear system");
        std::swap(matrix[col], matrix[pivot]);
        std::swap(rhs[col], rhs[pivot]);
        for (std::size_t row = col + 1; row < n; ++row) {
            const double factor = matrix[row][col] / matrix[col][col];
            matrix[row][col] = 0.0;
            for (std::size_t j = col + 1; j < n; ++j) matrix[row][j] -= factor * matrix[col][j];
            rhs[row] -= factor * rhs[col];
        }
    }
    Vector result(n, 0.0);
    for (std::size_t i = n; i-- > 0;) {
        double residual = rhs[i];
        for (std::size_t j = i + 1; j < n; ++j) residual -= matrix[i][j] * result[j];
        result[i] = residual / matrix[i][i];
        require(std::isfinite(result[i]), "nonfinite solution of linear system");
    }
    return result;
}

Allocation optimise_supports(const Moments &stats, const Matrix &covariance,
                            bool constrained, double target) {
    const std::size_t n = stats.mean.size();
    Allocation best;
    best.estimated_daily_variance = std::numeric_limits<double>::infinity();
    double covariance_scale = 0.0;
    for (std::size_t i = 0; i < n; ++i) covariance_scale = std::max(covariance_scale, covariance[i][i]);
    require(covariance_scale > 0.0, "covariance must have positive diagonal scale");
    double return_scale = 1e-5;
    for (double mean : stats.mean) return_scale = std::max(return_scale, std::abs(mean));

    for (unsigned mask = 1; mask < (1u << n); ++mask) {
        std::vector<std::size_t> support;
        for (std::size_t i = 0; i < n; ++i) if ((mask >> i) & 1u) support.push_back(i);
        const std::size_t k = support.size();
        Vector subset(k, 1.0);
        bool use_target = constrained;
        if (constrained) {
            double lowest = stats.mean[support[0]], highest = lowest;
            for (auto i : support) {
                lowest = std::min(lowest, stats.mean[i]);
                highest = std::max(highest, stats.mean[i]);
            }
            if (target < lowest - 1e-12 || target > highest + 1e-12) continue;
            if (highest - lowest < 1e-12) {
                if (std::abs(target - lowest) > 1e-12) continue;
                use_target = false;
            }
        }
        const std::size_t constraints = use_target ? 2 : 1;
        Matrix system(k + constraints, Vector(k + constraints, 0.0));
        Vector rhs(k + constraints, 0.0);
        for (std::size_t i = 0; i < k; ++i) {
            for (std::size_t j = 0; j < k; ++j)
                system[i][j] = covariance[support[i]][support[j]] / covariance_scale;
            system[i][k] = system[k][i] = 1.0;
            if (use_target) system[i][k + 1] = system[k + 1][i] = stats.mean[support[i]] / return_scale;
        }
        rhs[k] = 1.0;
        if (use_target) rhs[k + 1] = target / return_scale;
        try {
            subset = solve(system, rhs);
        } catch (const std::invalid_argument &) {
            continue; // A lower-dimensional support may still attain this boundary.
        }
        Vector weights(n, 0.0);
        bool feasible = true;
        for (std::size_t i = 0; i < k; ++i) {
            if (subset[i] < -1e-9) feasible = false;
            weights[support[i]] = std::max(0.0, subset[i]);
        }
        if (!feasible) continue;
        const double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
        if (std::abs(sum - 1.0) > 1e-8) continue;
        for (double &weight : weights) weight /= sum;
        const double mean = dot(weights, stats.mean);
        if (constrained && std::abs(mean - target) > 1e-9) continue;
        const double variance = quadratic_form(weights, covariance);
        if (variance < best.estimated_daily_variance) {
            best = {weights, mean, variance, constrained ? target : mean};
        }
    }
    require(!best.weights.empty(), "no feasible long-only allocation found");
    return best;
}

Matrix cholesky(const Matrix &covariance) {
    const std::size_t n = covariance.size();
    Matrix lower(n, Vector(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double value = covariance[i][j];
            for (std::size_t k = 0; k < j; ++k) value -= lower[i][k] * lower[j][k];
            if (i == j) {
                require(value > 0.0 && std::isfinite(value), "simulation covariance is not positive definite");
                lower[i][j] = std::sqrt(value);
            } else {
                lower[i][j] = value / lower[j][j];
            }
        }
    }
    return lower;
}

void validate_moments(const Moments &stats) {
    const std::size_t n = stats.mean.size();
    require(n >= 1 && n <= 8 && stats.covariance.size() == n, "invalid moment dimensions");
    for (std::size_t i = 0; i < n; ++i) {
        require(stats.covariance[i].size() == n && std::isfinite(stats.mean[i]), "invalid moment dimensions or mean");
        for (double value : stats.covariance[i]) require(std::isfinite(value), "nonfinite covariance");
    }
}

} // namespace

Prices read_prices(const std::string &path) {
    std::ifstream stream(path);
    require(stream.good(), "cannot open price CSV: " + path);
    std::string line;
    require(static_cast<bool>(std::getline(stream, line)), "price CSV is empty");
    auto header = csv_row(line);
    require(header.size() >= 2 && header.size() <= 9, "price CSV must contain one to eight assets");
    require(header[0] == "Date" || header[0] == "date", "first CSV column must be Date or date");
    Prices prices;
    std::set<std::string> names;
    for (std::size_t i = 1; i < header.size(); ++i) {
        require(!header[i].empty() && header[i].size() <= 64, "asset symbols must have one to 64 characters");
        for (unsigned char c : header[i]) require(c >= 32 && c < 127, "asset symbols must be printable ASCII");
        require(names.insert(header[i]).second, "duplicate asset symbol: " + header[i]);
        prices.assets.push_back(header[i]);
    }
    std::size_t row_number = 1;
    while (std::getline(stream, line)) {
        ++row_number;
        const auto fields = csv_row(line);
        require(fields.size() == header.size(), "CSV row " + std::to_string(row_number) + " has the wrong number of columns");
        require(valid_date(fields[0]), "invalid ISO date in row " + std::to_string(row_number));
        require(prices.dates.empty() || fields[0] > prices.dates.back(), "dates must be ascending and unique");
        Vector values;
        for (std::size_t i = 1; i < fields.size(); ++i) values.push_back(parse_price(fields[i]));
        prices.dates.push_back(fields[0]);
        prices.values.push_back(values);
        require(prices.values.size() <= 1000000, "price CSV exceeds one million rows");
    }
    require(prices.values.size() >= 3, "at least three price rows are required");
    return prices;
}

Returns simple_returns(const Prices &prices) {
    require(prices.values.size() >= 3, "at least three price rows are required");
    Returns returns;
    for (std::size_t t = 1; t < prices.values.size(); ++t) {
        Vector row;
        for (std::size_t i = 0; i < prices.assets.size(); ++i) {
            const double value = prices.values[t][i] / prices.values[t - 1][i] - 1.0;
            require(std::isfinite(value) && value > -1.0, "unrepresentable simple return in prices");
            row.push_back(value);
        }
        returns.values.push_back(row);
        returns.dates.push_back(prices.dates[t]);
    }
    return returns;
}

void validate_weights(const Vector &weights, std::size_t assets) {
    require(weights.size() == assets, "weight count must equal the number of assets");
    for (double weight : weights) require(std::isfinite(weight) && weight >= 0.0 && weight <= 1.0,
                                          "weights must be finite and between zero and one");
    const double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    require(std::abs(sum - 1.0) <= 1e-8, "weights must sum to one");
}

Moments moments(const Matrix &observations) {
    require(observations.size() >= 2 && !observations[0].empty(), "at least two nonempty observations are required");
    const std::size_t n = observations[0].size();
    Moments result{Vector(n, 0.0), Matrix(n, Vector(n, 0.0))};
    // Multivariate Welford avoids subtracting large raw second moments.
    std::size_t count = 0;
    for (const auto &row : observations) {
        require(row.size() == n, "inconsistent observation dimensions");
        ++count;
        Vector delta(n), residual(n);
        for (std::size_t i = 0; i < n; ++i) {
            require(std::isfinite(row[i]), "nonfinite observation");
            delta[i] = row[i] - result.mean[i];
            result.mean[i] += delta[i] / static_cast<double>(count);
            residual[i] = row[i] - result.mean[i];
        }
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j <= i; ++j) result.covariance[i][j] += delta[i] * residual[j];
    }
    for (std::size_t i = 0; i < n; ++i) {
        require(std::isfinite(result.mean[i]), "mean overflow");
        for (std::size_t j = 0; j <= i; ++j) {
            result.covariance[i][j] /= static_cast<double>(count - 1);
            require(std::isfinite(result.covariance[i][j]), "covariance overflow");
            result.covariance[j][i] = result.covariance[i][j];
        }
    }
    return result;
}

double dot(const Vector &left, const Vector &right) {
    require(left.size() == right.size(), "incompatible vector dimensions");
    return std::inner_product(left.begin(), left.end(), right.begin(), 0.0);
}

Vector portfolio_returns(const Matrix &returns, const Vector &weights) {
    require(!returns.empty(), "return sample cannot be empty");
    validate_weights(weights, returns[0].size());
    Vector portfolio;
    for (const auto &row : returns) {
        const double value = dot(weights, row);
        require(std::isfinite(value) && value > -1.0, "portfolio returns must be finite and above -1");
        portfolio.push_back(value);
    }
    return portfolio;
}

double quantile(Vector values, double probability) {
    require(!values.empty() && std::isfinite(probability) && probability >= 0.0 && probability <= 1.0,
            "quantile requires a nonempty sample and probability in [0,1]");
    for (double value : values) require(std::isfinite(value), "nonfinite quantile observation");
    std::sort(values.begin(), values.end());
    const double position = probability * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = std::min(lower + 1, values.size() - 1);
    const double fraction = position - static_cast<double>(lower);
    return (1.0 - fraction) * values[lower] + fraction * values[upper];
}

double expected_shortfall(Vector losses, double confidence) {
    confidence_valid(confidence);
    require(!losses.empty(), "expected shortfall requires observations");
    for (double loss : losses) require(std::isfinite(loss), "nonfinite loss");
    std::sort(losses.begin(), losses.end(), std::greater<double>());
    const double mass = (1.0 - confidence) * static_cast<double>(losses.size());
    const auto full = static_cast<std::size_t>(std::floor(mass));
    const double fractional = mass - static_cast<double>(full);
    double sum = std::accumulate(losses.begin(), losses.begin() + static_cast<std::ptrdiff_t>(full), 0.0);
    if (fractional > 0.0 && full < losses.size()) sum += fractional * losses[full];
    return sum / mass;
}

Metrics metrics(const Vector &returns, double confidence) {
    confidence_valid(confidence);
    require(returns.size() >= 2, "risk metrics require at least two returns");
    Matrix observations;
    Vector losses;
    double log_wealth = 0.0;
    for (double value : returns) {
        require(std::isfinite(value) && value > -1.0, "returns must be finite and greater than -1");
        observations.push_back({value});
        losses.push_back(-value);
        log_wealth += std::log1p(value);
    }
    const auto stats = moments(observations);
    const double volatility = std::sqrt(std::max(0.0, stats.covariance[0][0]));
    const double total = std::expm1(log_wealth);
    const double annual = std::expm1(log_wealth * 252.0 / static_cast<double>(returns.size()));
    require(std::isfinite(total) && std::isfinite(annual), "compounded return overflow");
    return {returns.size(), stats.mean[0], volatility, volatility * std::sqrt(252.0), total,
            annual, quantile(losses, confidence), expected_shortfall(losses, confidence)};
}

double quadratic_form(const Vector &weights, const Matrix &covariance) {
    require(covariance.size() == weights.size(), "incompatible covariance dimensions");
    double variance = 0.0;
    for (std::size_t i = 0; i < weights.size(); ++i) variance += weights[i] * dot(covariance[i], weights);
    require(std::isfinite(variance), "portfolio variance overflow");
    return variance;
}

VarianceContributions variance_contributions(const Vector &weights, const Matrix &covariance) {
    validate_weights(weights, covariance.size());
    const double total = quadratic_form(weights, covariance);
    require(total >= -1e-14, "portfolio variance must be nonnegative");
    VarianceContributions result;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        const double value = weights[i] * dot(covariance[i], weights);
        result.absolute_daily_variance.push_back(value);
        if (total > 0.0) result.fractional.push_back(value / total);
    }
    return result;
}

Matrix add_ridge(Matrix covariance, double ridge) {
    require(std::isfinite(ridge) && ridge > 0.0, "ridge must be finite and positive");
    for (std::size_t i = 0; i < covariance.size(); ++i) covariance[i][i] += ridge;
    return covariance;
}

Allocation minimum_variance(const Moments &stats, double ridge) {
    validate_moments(stats);
    return optimise_supports(stats, add_ridge(stats.covariance, ridge), false, 0.0);
}

std::vector<Allocation> efficient_frontier(const Moments &stats, std::size_t points, double ridge) {
    require(points >= 2 && points <= 100, "frontier-points must be between two and 100");
    validate_moments(stats);
    const auto covariance = add_ridge(stats.covariance, ridge);
    const auto minimum = optimise_supports(stats, covariance, false, 0.0);
    const double highest = *std::max_element(stats.mean.begin(), stats.mean.end());
    std::vector<Allocation> frontier;
    for (std::size_t i = 0; i < points; ++i) {
        const double target = minimum.estimated_daily_return +
            (highest - minimum.estimated_daily_return) * static_cast<double>(i) / static_cast<double>(points - 1);
        frontier.push_back(optimise_supports(stats, covariance, true, target));
    }
    return frontier;
}

Simulation monte_carlo(const Prices &prices, const Vector &weights, std::size_t paths,
                       std::size_t horizon, uint64_t seed) {
    require(paths >= 100 && paths <= 1000000, "paths must be between 100 and 1000000");
    require(horizon >= 1 && horizon <= 20, "horizon must be between one and 20");
    validate_weights(weights, prices.assets.size());
    Matrix log_returns;
    for (std::size_t t = 1; t < prices.values.size(); ++t) {
        Vector row;
        for (std::size_t i = 0; i < weights.size(); ++i)
            row.push_back(std::log(prices.values[t][i]) - std::log(prices.values[t - 1][i]));
        log_returns.push_back(row);
    }
    Simulation result;
    result.log_moments = moments(log_returns);
    result.effective_log_covariance = add_ridge(result.log_moments.covariance, simulation_ridge);
    const auto lower = cholesky(result.effective_log_covariance);
    qfa_rng rng;
    qfa_rng_seed(&rng, seed);
    result.portfolio_returns.reserve(paths);
    for (std::size_t path = 0; path < paths; ++path) {
        double wealth = 1.0;
        for (std::size_t day = 0; day < horizon; ++day) {
            Vector normals(weights.size());
            for (double &value : normals) value = qfa_rng_normal(&rng);
            double gross_return = 0.0;
            for (std::size_t i = 0; i < weights.size(); ++i) {
                double log_return = result.log_moments.mean[i];
                for (std::size_t j = 0; j <= i; ++j) log_return += lower[i][j] * normals[j];
                gross_return += weights[i] * std::exp(log_return);
            }
            wealth *= gross_return;
            require(std::isfinite(wealth) && wealth > 0.0, "simulation wealth overflow or underflow");
        }
        result.portfolio_returns.push_back(wealth - 1.0);
    }
    return result;
}

std::vector<Forecast> rolling_var(const Returns &returns, const Vector &weights,
                                 std::size_t window, double confidence) {
    confidence_valid(confidence);
    require(window >= 2 && window < returns.values.size(), "window must be at least two and shorter than the return history");
    require(returns.dates.size() == returns.values.size(), "return dates must match values");
    const auto portfolio = portfolio_returns(returns.values, weights);
    std::vector<Forecast> forecasts;
    for (std::size_t t = window; t < portfolio.size(); ++t) {
        Vector losses;
        losses.reserve(window);
        for (std::size_t j = t - window; j < t; ++j) losses.push_back(-portfolio[j]);
        const double var = quantile(losses, confidence);
        const double loss = -portfolio[t];
        forecasts.push_back({returns.dates[t], var, loss, loss > var});
    }
    return forecasts;
}

Kupiec kupiec(const std::vector<Forecast> &forecasts, double expected_probability) {
    confidence_valid(expected_probability);
    require(!forecasts.empty(), "Kupiec test requires forecasts");
    std::size_t count = 0;
    for (const auto &forecast : forecasts) if (forecast.breach) ++count;
    const double n = static_cast<double>(forecasts.size());
    const double x = static_cast<double>(count);
    const double observed = x / n;
    const double expected_log_likelihood = x * std::log(expected_probability) +
        (n - x) * std::log1p(-expected_probability);
    double observed_log_likelihood = 0.0;
    if (count > 0 && count < forecasts.size())
        observed_log_likelihood = x * std::log(observed) + (n - x) * std::log1p(-observed);
    const double lr = std::max(0.0, 2.0 * (observed_log_likelihood - expected_log_likelihood));
    return {forecasts.size(), count, expected_probability, observed, lr,
            std::erfc(std::sqrt(lr / 2.0))};
}

std::vector<BootstrapInterval> paired_block_bootstrap(
    const Vector &input_returns, const Vector &candidate_returns, double confidence,
    const std::vector<std::size_t> &block_sizes, std::size_t repetitions, uint64_t seed) {
    confidence_valid(confidence);
    const std::size_t n = input_returns.size();
    require(n >= 2 && candidate_returns.size() == n, "paired bootstrap requires equal-length samples of at least two returns");
    require(repetitions >= 2 && repetitions <= 100000, "bootstrap repetitions must be between two and 100000");
    auto summary = [confidence](const Vector &returns) {
        Vector losses;
        losses.reserve(returns.size());
        double mean = 0.0, second = 0.0;
        std::size_t count = 0;
        for (double value : returns) {
            require(std::isfinite(value) && value > -1.0, "bootstrap returns must be finite and greater than -1");
            ++count;
            const double delta = value - mean;
            mean += delta / static_cast<double>(count);
            second += delta * (value - mean);
            losses.push_back(-value);
        }
        return Vector{252.0 * second / static_cast<double>(returns.size() - 1),
                      quantile(losses, confidence), expected_shortfall(losses, confidence)};
    };
    const Vector input_summary = summary(input_returns);
    const Vector candidate_summary = summary(candidate_returns);
    const std::vector<std::string> names = {"annualized_variance", "historical_var", "historical_es"};
    std::vector<BootstrapInterval> intervals;
    for (std::size_t block_size : block_sizes) {
        require(block_size >= 1 && block_size <= n, "bootstrap block size must be between one and sample length");
        qfa_rng rng;
        qfa_rng_seed(&rng, seed);
        Matrix differences(names.size());
        for (auto &series : differences) series.reserve(repetitions);
        for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
            Vector sampled_input, sampled_candidate;
            sampled_input.reserve(n);
            sampled_candidate.reserve(n);
            while (sampled_input.size() < n) {
                const auto start = static_cast<std::size_t>(qfa_rng_bounded(&rng, n - block_size + 1));
                for (std::size_t offset = 0; offset < block_size && sampled_input.size() < n; ++offset) {
                    sampled_input.push_back(input_returns[start + offset]);
                    sampled_candidate.push_back(candidate_returns[start + offset]);
                }
            }
            const auto input = summary(sampled_input);
            const auto candidate = summary(sampled_candidate);
            for (std::size_t metric = 0; metric < names.size(); ++metric)
                differences[metric].push_back(input[metric] - candidate[metric]);
        }
        for (std::size_t metric = 0; metric < names.size(); ++metric) {
            intervals.push_back({names[metric], block_size,
                                 input_summary[metric] - candidate_summary[metric],
                                 quantile(differences[metric], 0.025),
                                 quantile(differences[metric], 0.975)});
        }
    }
    return intervals;
}

} // namespace qfa
