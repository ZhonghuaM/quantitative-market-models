#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

struct MonteCarloResult {
    double price;
    double standard_error;
};

MonteCarloResult price_european_call(
    double spot,
    double strike,
    double rate,
    double volatility,
    double maturity,
    int paths,
    unsigned int seed
) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<double> discounted_payoffs;
    discounted_payoffs.reserve(paths);

    const double drift = (rate - 0.5 * volatility * volatility) * maturity;
    const double diffusion = volatility * std::sqrt(maturity);
    const double discount = std::exp(-rate * maturity);

    for (int i = 0; i < paths; ++i) {
        const double terminal = spot * std::exp(drift + diffusion * normal(rng));
        discounted_payoffs.push_back(discount * std::max(terminal - strike, 0.0));
    }

    const double mean = std::accumulate(discounted_payoffs.begin(), discounted_payoffs.end(), 0.0) / paths;
    double squared_error = 0.0;
    for (double payoff : discounted_payoffs) {
        squared_error += (payoff - mean) * (payoff - mean);
    }
    const double variance = squared_error / static_cast<double>(paths - 1);
    return {mean, std::sqrt(variance / paths)};
}

int main() {
    const auto result = price_european_call(100.0, 100.0, 0.05, 0.20, 1.0, 500000, 42);
    std::cout << std::fixed << std::setprecision(6)
              << "Monte Carlo European call price: " << result.price << "\n"
              << "Standard error: " << result.standard_error << "\n";
    return 0;
}
