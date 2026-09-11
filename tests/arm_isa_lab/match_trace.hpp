#pragma once
#include "nss/cpu_api.hpp"
#include <cstdint>

std::uint64_t nss_trace_hash(const float* p, int width, int height, int stride);
int nss_trace_matches(int kind, int x, int y, int block, int capacity, int width, int height,
                       std::uint64_t input_hash, nss::Match* matches, int count);
void nss_trace_indices(int m, int n, int q, std::uint64_t input_hash, int* indices);
