/* SplitMix64 and a cached Box-Muller transform; no platform rand() state. */
#include "qfa_rng.h"

#include <math.h>

void qfa_rng_seed(qfa_rng *rng, uint64_t seed) {
    rng->state = seed;
    rng->has_spare = 0;
    rng->spare = 0.0;
}

uint64_t qfa_rng_u64(qfa_rng *rng) {
    uint64_t z = (rng->state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

uint64_t qfa_rng_bounded(qfa_rng *rng, uint64_t bound) {
    /* Rejection removes modulo bias. The caller supplies a positive bound. */
    if (bound == 0) return 0;
    const uint64_t threshold = (UINT64_C(0) - bound) % bound;
    uint64_t value;
    do { value = qfa_rng_u64(rng); } while (value < threshold);
    return value % bound;
}

double qfa_rng_uniform(qfa_rng *rng) {
    /* 52 random bits plus half a bin: both endpoints are excluded exactly. */
    return ((double)(qfa_rng_u64(rng) >> 12) + 0.5) * 0x1.0p-52;
}

double qfa_rng_normal(qfa_rng *rng) {
    if (rng->has_spare) {
        rng->has_spare = 0;
        return rng->spare;
    }
    const double radius = sqrt(-2.0 * log(qfa_rng_uniform(rng)));
    const double angle = 6.283185307179586476925286766559 * qfa_rng_uniform(rng);
    rng->spare = radius * sin(angle);
    rng->has_spare = 1;
    return radius * cos(angle);
}
