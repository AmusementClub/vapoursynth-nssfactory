#pragma once

#include "nss/resources.hpp"
#include "nss/params.hpp"

namespace nss {
struct NlhGroupOptions {
    int q = 4;
    double sigma = 0;
    bool wiener = false;
    double hard_strength = 1;
    int wiener_iterations = 2;
    double wiener_sigma_scale = kNlhWienerSigmaScale;
};
struct NlhWorkspace {
    ResourceVector<int> indices;
    ResourceVector<float> matrix, reference;
    ResourceVector<double> numerator, denominator;
};

// Full q x n pixel-matrix aggregation. n must be a supported power of two,
// including one for an image boundary with only its reference patch available.
// row_indices optionally supplies m*q shared guide indices (self first).
// Output is an unnormalized contribution for every original group element;
// the host must accumulate numerator AND counts before image normalization.
void nlh_filter_full(const float* input, const float* reference, int m, int n, int lda,
                     const NlhGroupOptions& options, const int* row_indices, NlhWorkspace& work);
struct NlhFullBatchItem {
    const float* input = nullptr;
    const float* reference = nullptr;
    int m = 0, n = 0, lda = 0;
    const NlhGroupOptions* options = nullptr;
    const int* row_indices = nullptr;
    NlhWorkspace* work = nullptr;
};
void nlh_filter_full_batch(NlhFullBatchItem* items, int count);

// Two-sided normalized Haar. The optimized q4/n16 implementation is shared
// with the public 1D transform; every other power-of-two shape is supported.
void nlh_haar2d(float* matrix, int q, int n, bool inverse);
} // namespace nss
