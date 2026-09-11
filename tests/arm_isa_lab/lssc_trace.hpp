#pragma once
#include <cstdint>

// Test-only hooks. The generator inserts calls into a private source copy;
// the production object and its public ABI are not modified.
void nss_trace_omp(const float* y, int m, const float* dictionary, int atoms, int ldd,
                   const float* residual, const float* correlations, const char* used,
                   int selected, float magnitude, int step);
