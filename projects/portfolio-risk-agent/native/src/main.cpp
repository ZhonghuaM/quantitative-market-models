#include "qfa.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

constexpr const char *version = "1.0.0";

std::string quote(const std::string &value) {
    std::ostringstream stream;
    stream << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') stream << '\\' << c;
        else if (c < 32) stream << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned>(c) << std::dec;
        else stream << c;
    }
    stream << '"';
    return stream.str();
}

std::string number(double value) {
    if (!std::isfinite(value)) throw std::runtime_error("nonfinite numerical result");
    std::ostringstream stream;
    stream << std::setprecision(17) << value;
    return stream.str();
}

std::string array(const qfa::Vector &values) {
    std::string result = "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) result += ',';
        result += number(values[i]);
    }
    return result + "]";
}

std::string array(const std::vector<std::string> &values) {
    std::string result = "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) result += ',';
        result += quote(values[i]);
    }
    return result + "]";
}

std::string array(const qfa::Matrix &values) {
    std::string result = "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) result += ',';
        result += array(values[i]);
    }
    return result + "]";
}

class Object {
    std::string value_ = "{";
    bool first_ = true;
public:
    void raw(const std::string &key, const std::string &value) {
        if (!first_) value_ += ',';
        first_ = false;
        value_ += quote(key) + ':' + value;
    }
    void text(const std::string &key, const std::string &value) { raw(key, quote(value)); }
    void numeric(const std::string &key, double value) { raw(key, number(value)); }
    void integer(const std::string &key, uint64_t value) { raw(key, std::to_string(value)); }
    std::string json() const { return value_ + '}'; }
};

void append_metrics(Object &output, const qfa::Metrics &metrics) {
    output.integer("observations", metrics.observations);
    output.numeric("mean_daily_return", metrics.mean_daily_return);
    output.numeric("volatility_daily", metrics.volatility_daily);
    output.numeric("annualized_volatility", metrics.annualized_volatility);
    output.numeric("total_return", metrics.total_return);
    output.numeric("annualized_return", metrics.annualized_return);
    output.numeric("historical_var", metrics.historical_var);
    output.numeric("historical_es", metrics.historical_es);
}

std::string metrics_json(const qfa::Matrix &returns, const qfa::Vector &weights, double confidence) {
    Object result;
    append_metrics(result, qfa::metrics(qfa::portfolio_returns(returns, weights), confidence));
    return result.json();
}

std::string allocation_json(const qfa::Allocation &allocation) {
    Object result;
    result.raw("weights", array(allocation.weights));
    result.numeric("estimated_daily_return", allocation.estimated_daily_return);
    result.numeric("estimated_daily_variance", allocation.estimated_daily_variance);
    result.numeric("target_daily_return", allocation.target_daily_return);
    return result.json();
}

std::string contributions_json(const qfa::Vector &weights, const qfa::Matrix &covariance) {
    const auto contributions = qfa::variance_contributions(weights, covariance);
    Object result;
    result.raw("absolute_daily_variance", array(contributions.absolute_daily_variance));
    result.raw("fractional", contributions.fractional.empty() ? "null" : array(contributions.fractional));
    return result.json();
}

double parse_double(const std::string &text, const std::string &name) {
    if (text.empty()) throw std::invalid_argument(name + " is missing a numeric value");
    errno = 0;
    char *end = nullptr;
    const double value = std::strtod(text.c_str(), &end);
    if (end != text.c_str() + text.size() || errno == ERANGE || !std::isfinite(value))
        throw std::invalid_argument(name + " must be a finite number");
    return value;
}

uint64_t parse_integer(const std::string &text, const std::string &name) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument(name + " must be an unsigned integer");
    std::size_t position = 0;
    try {
        const auto value = std::stoull(text, &position);
        if (position != text.size()) throw std::invalid_argument("trailing characters");
        return value;
    } catch (const std::exception &) {
        throw std::invalid_argument(name + " is outside the unsigned 64-bit range");
    }
}

qfa::Vector parse_weights(const std::string &text) {
    qfa::Vector weights;
    std::size_t start = 0;
    while (start <= text.size()) {
        auto end = text.find(',', start);
        if (end == std::string::npos) end = text.size();
        weights.push_back(parse_double(text.substr(start, end - start), "weight"));
        if (end == text.size()) break;
        start = end + 1;
    }
    return weights;
}

std::map<std::string, std::string> arguments(int argc, char **argv, const std::string &command) {
    std::set<std::string> allowed = {"--prices", "--weights", "--confidence"};
    if (command == "optimise") allowed.insert({"--train", "--frontier-points"});
    if (command == "monte-carlo") allowed.insert({"--paths", "--horizon", "--seed"});
    if (command == "backtest") allowed.insert("--window");
    std::map<std::string, std::string> values;
    for (int i = 2; i < argc; i += 2) {
        const std::string flag = argv[i];
        if (!allowed.count(flag)) throw std::invalid_argument("unknown or inapplicable flag: " + flag);
        if (values.count(flag)) throw std::invalid_argument("duplicate flag: " + flag);
        if (i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0)
            throw std::invalid_argument("missing value for " + flag);
        values[flag] = argv[i + 1];
    }
    if (!values.count("--prices") || !values.count("--weights"))
        throw std::invalid_argument("--prices and --weights are required");
    return values;
}

uint64_t integer_option(const std::map<std::string, std::string> &options,
                        const std::string &flag, uint64_t fallback) {
    const auto found = options.find(flag);
    return found == options.end() ? fallback : parse_integer(found->second, flag);
}

Object period_json(const std::vector<std::string> &dates, std::size_t begin, std::size_t end) {
    Object result;
    result.integer("rows", end - begin);
    result.text("start_date", dates.at(begin));
    result.text("end_date", dates.at(end - 1));
    return result;
}

int run(int argc, char **argv) {
    if (argc < 2) throw std::invalid_argument("usage: qfa_engine risk|optimise|monte-carlo|backtest --prices CSV --weights CSV [options]");
    const std::string command = argv[1];
    if (command == "version" || command == "--version") {
        if (argc != 2) throw std::invalid_argument("version takes no options");
        Object output;
        output.integer("schema_version", 1);
        output.text("native_version", version);
        output.text("compiler", __VERSION__);
        output.text("language", "C17/C++17");
        output.raw("ieee754_double", std::numeric_limits<double>::is_iec559 ? "true" : "false");
        std::cout << output.json() << '\n';
        return 0;
    }
    if (command != "risk" && command != "optimise" && command != "monte-carlo" && command != "backtest")
        throw std::invalid_argument("unknown command: " + command);
    const auto options = arguments(argc, argv, command);
    const double confidence = options.count("--confidence") ? parse_double(options.at("--confidence"), "confidence") : 0.99;
    if (!(confidence > 0.0 && confidence < 1.0)) throw std::invalid_argument("confidence must be strictly between zero and one");
    const auto weights = parse_weights(options.at("--weights"));
    const auto prices = qfa::read_prices(options.at("--prices"));
    qfa::validate_weights(weights, prices.assets.size());
    const auto returns = qfa::simple_returns(prices);
    Object output, settings, data;
    output.integer("schema_version", 1);
    output.text("native_version", version);
    output.text("command", command);
    output.raw("assets", array(prices.assets));
    output.raw("weights", array(weights));
    settings.numeric("confidence", confidence);
    data.integer("price_rows", prices.values.size());
    data.integer("return_rows", returns.values.size());
    data.text("start_date", prices.dates.front());
    data.text("end_date", prices.dates.back());
    output.raw("data", data.json());

    if (command == "risk") {
        append_metrics(output, qfa::metrics(qfa::portfolio_returns(returns.values, weights), confidence));
        const auto stats = qfa::moments(returns.values);
        output.raw("asset_mean_daily_returns", array(stats.mean));
        output.raw("sample_covariance", array(stats.covariance));
    } else if (command == "optimise") {
        const auto train = integer_option(options, "--train", 504);
        const auto points = integer_option(options, "--frontier-points", 15);
        if (train < 2 || train > returns.values.size() - 2)
            throw std::invalid_argument("train must leave at least two training and two holdout returns");
        if (points < 2 || points > 100) throw std::invalid_argument("frontier-points must be between two and 100");
        settings.integer("train", train);
        settings.integer("frontier_points", points);
        const qfa::Matrix training(returns.values.begin(), returns.values.begin() + static_cast<std::ptrdiff_t>(train));
        const qfa::Matrix holdout(returns.values.begin() + static_cast<std::ptrdiff_t>(train), returns.values.end());
        const auto stats = qfa::moments(training);
        const auto holdout_stats = qfa::moments(holdout);
        const auto minimum = qfa::minimum_variance(stats);
        const auto frontier = qfa::efficient_frontier(stats, points);
        output.raw("training", period_json(returns.dates, 0, train).json());
        output.raw("holdout", period_json(returns.dates, train, returns.values.size()).json());
        output.numeric("ridge", qfa::optimisation_ridge);
        output.raw("asset_mean_daily_returns", array(stats.mean));
        output.raw("sample_covariance", array(stats.covariance));
        output.raw("minimum_variance", allocation_json(minimum));
        std::string curve = "[";
        for (std::size_t i = 0; i < frontier.size(); ++i) {
            if (i) curve += ',';
            curve += allocation_json(frontier[i]);
        }
        output.raw("frontier", curve + ']');
        Object comparison;
        for (const auto &entry : std::map<std::string, qfa::Vector>{{"input", weights}, {"minimum_variance", minimum.weights}}) {
            Object result;
            result.raw("weights", array(entry.second));
            result.raw("training", metrics_json(training, entry.second, confidence));
            result.raw("holdout", metrics_json(holdout, entry.second, confidence));
            result.raw("training_variance_contributions", contributions_json(entry.second, stats.covariance));
            result.raw("holdout_variance_contributions", contributions_json(entry.second, holdout_stats.covariance));
            comparison.raw(entry.first, result.json());
        }
        output.raw("comparison", comparison.json());
        std::vector<std::size_t> block_sizes;
        qfa::Vector skipped;
        for (std::size_t block : {10u, 20u}) {
            if (block <= holdout.size()) block_sizes.push_back(block);
            else skipped.push_back(static_cast<double>(block));
        }
        const auto intervals = qfa::paired_block_bootstrap(
            qfa::portfolio_returns(holdout, weights),
            qfa::portfolio_returns(holdout, minimum.weights), confidence, block_sizes, 1000, 42);
        std::string interval_rows = "[";
        for (std::size_t i = 0; i < intervals.size(); ++i) {
            if (i) interval_rows += ',';
            Object interval;
            interval.text("metric", intervals[i].metric);
            interval.integer("block_size", intervals[i].block_size);
            interval.numeric("estimate", intervals[i].estimate);
            interval.numeric("ci_lower", intervals[i].ci_lower);
            interval.numeric("ci_upper", intervals[i].ci_upper);
            interval_rows += interval.json();
        }
        output.raw("holdout_uncertainty", interval_rows + ']');
        output.raw("holdout_uncertainty_skipped_block_sizes", array(skipped));
        Object uncertainty_settings;
        uncertainty_settings.integer("repetitions", 1000);
        uncertainty_settings.integer("seed", 42);
        uncertainty_settings.raw("block_sizes", "[10,20]");
        uncertainty_settings.numeric("confidence_level", 0.95);
        uncertainty_settings.text("method", "paired moving-block percentile; fixed allocations; input minus candidate");
        output.raw("holdout_uncertainty_settings", uncertainty_settings.json());
    } else if (command == "monte-carlo") {
        const auto paths = integer_option(options, "--paths", 20000);
        const auto horizon = integer_option(options, "--horizon", 1);
        const auto seed = integer_option(options, "--seed", 42);
        if (paths < 100 || paths > 1000000) throw std::invalid_argument("paths must be between 100 and 1000000");
        if (horizon < 1 || horizon > 20) throw std::invalid_argument("horizon must be between one and 20");
        settings.integer("paths", paths);
        settings.integer("horizon", horizon);
        settings.integer("seed", seed);
        const auto simulation = qfa::monte_carlo(prices, weights, paths, horizon, seed);
        qfa::Vector losses;
        qfa::Matrix observations;
        for (double value : simulation.portfolio_returns) {
            losses.push_back(-value);
            observations.push_back({value});
        }
        const auto stats = qfa::moments(observations);
        output.numeric("monte_carlo_var", qfa::quantile(losses, confidence));
        output.numeric("monte_carlo_es", qfa::expected_shortfall(losses, confidence));
        output.numeric("mean_simulated_return", stats.mean[0]);
        output.numeric("simulated_return_std", std::sqrt(std::max(0.0, stats.covariance[0][0])));
        output.raw("estimated_log_mean", array(simulation.log_moments.mean));
        output.raw("estimated_log_covariance", array(simulation.log_moments.covariance));
        output.raw("effective_log_covariance", array(simulation.effective_log_covariance));
        output.numeric("covariance_ridge", qfa::simulation_ridge);
        output.text("generator", "SplitMix64; 52-bit open-interval uniforms; cached Box-Muller");
    } else {
        const auto window = integer_option(options, "--window", 250);
        settings.integer("window", window);
        settings.integer("horizon", 1);
        const auto forecasts = qfa::rolling_var(returns, weights, window, confidence);
        std::string rows = "[";
        for (std::size_t i = 0; i < forecasts.size(); ++i) {
            if (i) rows += ',';
            Object forecast;
            forecast.text("date", forecasts[i].date);
            forecast.numeric("var", forecasts[i].var);
            forecast.numeric("loss", forecasts[i].loss);
            forecast.raw("breach", forecasts[i].breach ? "true" : "false");
            rows += forecast.json();
        }
        output.raw("forecasts", rows + ']');
        const auto test = qfa::kupiec(forecasts, 1.0 - confidence);
        Object kupiec;
        kupiec.integer("observations", test.observations);
        kupiec.integer("breaches", test.breaches);
        kupiec.numeric("expected_breach_rate", test.expected_breach_rate);
        kupiec.numeric("observed_breach_rate", test.observed_breach_rate);
        kupiec.numeric("lr_statistic", test.lr_statistic);
        kupiec.numeric("p_value", test.p_value);
        output.raw("kupiec", kupiec.json());
        output.text("forecast_timing", "Row t loss uses return ending at t; VaR uses exactly the window returns strictly before t.");
    }
    output.raw("settings", settings.json());
    std::cout << output.json() << '\n';
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    try {
        return run(argc, argv);
    } catch (const std::invalid_argument &error) {
        std::cout << "{\"error\":{\"code\":\"input_error\",\"message\":" << quote(error.what()) << "}}\n";
        return 2;
    } catch (const std::exception &error) {
        std::cout << "{\"error\":{\"code\":\"numerical_error\",\"message\":" << quote(error.what()) << "}}\n";
        return 3;
    }
}
