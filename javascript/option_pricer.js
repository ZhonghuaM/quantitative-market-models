function normCdf(x) {
  const sign = x < 0 ? -1 : 1;
  const z = Math.abs(x) / Math.sqrt(2);
  const t = 1 / (1 + 0.3275911 * z);
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const erf = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-z * z);
  return 0.5 * (1 + sign * erf);
}

function blackScholesCall(spot, strike, rate, volatility, maturity) {
  const sqrtT = Math.sqrt(maturity);
  const d1 = (Math.log(spot / strike) + (rate + 0.5 * volatility ** 2) * maturity) / (volatility * sqrtT);
  const d2 = d1 - volatility * sqrtT;
  return spot * normCdf(d1) - strike * Math.exp(-rate * maturity) * normCdf(d2);
}

function crrBinomialCall(spot, strike, rate, volatility, maturity, steps) {
  const dt = maturity / steps;
  const up = Math.exp(volatility * Math.sqrt(dt));
  const down = 1 / up;
  const probability = (Math.exp(rate * dt) - down) / (up - down);
  let values = [];
  for (let i = 0; i <= steps; i += 1) {
    const terminal = spot * up ** (steps - i) * down ** i;
    values.push(Math.max(terminal - strike, 0));
  }
  const discount = Math.exp(-rate * dt);
  for (let step = steps; step > 0; step -= 1) {
    const next = [];
    for (let i = 0; i < step; i += 1) {
      next.push(discount * (probability * values[i] + (1 - probability) * values[i + 1]));
    }
    values = next;
  }
  return values[0];
}

if (require.main === module) {
  const spot = 100;
  const strike = 100;
  const rate = 0.05;
  const volatility = 0.2;
  const maturity = 1;
  console.log(`Black-Scholes call: ${blackScholesCall(spot, strike, rate, volatility, maturity).toFixed(6)}`);
  console.log(`CRR 500-step call: ${crrBinomialCall(spot, strike, rate, volatility, maturity, 500).toFixed(6)}`);
}

module.exports = { blackScholesCall, crrBinomialCall };
