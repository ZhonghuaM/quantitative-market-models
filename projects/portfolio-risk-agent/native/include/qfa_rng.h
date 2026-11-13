#ifndef QFA_RNG_H
#define QFA_RNG_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t state;
    int has_spare;
    double spare;
} qfa_rng;

void qfa_rng_seed(qfa_rng *rng, uint64_t seed);
uint64_t qfa_rng_u64(qfa_rng *rng);
uint64_t qfa_rng_bounded(qfa_rng *rng, uint64_t bound);
double qfa_rng_uniform(qfa_rng *rng);
double qfa_rng_normal(qfa_rng *rng);

#ifdef __cplusplus
}
#endif
#endif
