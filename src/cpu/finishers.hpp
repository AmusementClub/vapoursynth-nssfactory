#pragma once

#include "nss/cpu_api.hpp"
#include "nss/cpu_common.hpp"
#include "nss/cpu_ncsr.hpp"
#include "cpu/wnnm/jacobi8.hpp"
#include <algorithm>
#include <cmath>

namespace nss::detail {
inline void finish_wnnm(float* group, int m, int n, int lda, float sigma, int residual, int adaptive,
                        float* adaptive_weight, float* U, float* S, float* Vt, const float* mean) {
    const float constant = 8.f * std::sqrt(2.f * static_cast<float>(n)) * sigma * sigma;
    const int kept = sv_shrink(S, std::min(m, n), constant, residual ? 0 : 1);
    if (adaptive_weight) *adaptive_weight = adaptive && kept > 0 ? 1.f / static_cast<float>(kept) : 1.f;
    if (m < n) {
        for (int col = 0; col < kept; ++col)
            for (int row = 0; row < m; ++row) U[row + col * m] *= S[col];
    } else {
        for (int col = 0; col < n; ++col)
            for (int row = 0; row < kept; ++row) Vt[row + col * n] *= S[row];
    }
    if (lda == m) {
        gemm_nn_hwy(m, n, kept, U, m, Vt, n, group, lda);
    } else {
        for (int col = 0; col < n; ++col) {
            for (int row = 0; row < m; ++row) {
                float sum = 0.f;
                for (int inner = 0; inner < kept; ++inner) sum += U[row + inner * m] * Vt[inner + col * n];
                group[row + col * lda] = sum;
            }
        }
    }
    if (residual) group_center_add(group, m, n, lda, mean);
}

inline void finish_pca_reconstruction(float* group, int m, int n, int lda, const float* U,
                                      const float* B, const float* mean) {
    const int r = std::min(m, n);
    gemm_nn_hwy(m, n, r, U, m, B, r, group, lda);
    group_center_add(group, m, n, lda, mean);
}
struct ScalarCodeRows {
    void gather(float* row, const float* B, int i, int stride, int count) const {
        for (int col = 0; col < count; ++col) row[col] = B[i + col * stride];
    }
    void scatter(float* B, int i, int stride, int count, const float* row) const {
        for (int col = 0; col < count; ++col) B[i + col * stride] = row[col];
    }
};

template<class Rows>
inline void finish_twsc_codes(int r, int n, float sigma, const float* col_sigma, float* col_weight,
                              float* S, float* B, const Rows& rows) {
    float sigmas[kSvdMaxN], thresholds[kSvdMaxN], row[kSvdMaxN];
    constexpr float epsilon = 1e-6f;
    bool same = true;
    for (int col = 0; col < n; ++col) {
        float value = col_sigma ? col_sigma[col] : sigma;
        if (!is_finite_bits(value) || value < 0.f) value = 0.f;
        sigmas[col] = value;
        if (col > 0 && value != sigmas[0]) same = false;
        if (col_weight) col_weight[col] = 1.f / (value + epsilon);
    }
    const float sigma0 = sigmas[0];
    const float noise = static_cast<float>(n) * sigma0 * sigma0;
    for (int i = 0; i < r; ++i) {
        const float singular = S[i];
        S[i] = std::sqrt(std::max(singular * singular - noise, 0.f));
        const float denominator = S[i] + epsilon;
        rows.gather(row, B, i, r, n);
        if (same) soft_threshold(row, n, sigma0 * sigma0 / denominator);
        else {
            for (int col = 0; col < n; ++col) thresholds[col] = sigmas[col] * sigmas[col] / denominator;
            soft_threshold_var(row, thresholds, n);
        }
        rows.scatter(B, i, r, n, row);
    }
}

template<class Rows, class Weights, class Centralize>
inline void finish_ncsr_codes(float* group, int m, int n, int lda, float sigma, const float* distance,
                              float* B, const Rows& rows, const Weights& make_weights, const Centralize& centralize) {
    if (!(sigma > 0.f) || !is_finite_bits(sigma)) return;
    const int r = std::min(m, n);
    float weights[kSvdMaxN], row_tau[kSvdMaxN], row[kSvdMaxN];
    constexpr float epsilon = 1e-12f;
    const float h = std::max(2.f * static_cast<float>(m) * sigma * sigma, epsilon);
    float sum = make_weights(distance, group, m, n, lda, h, weights);
    if (!(sum > 0.f)) { std::fill_n(weights, n, 1.f); sum = static_cast<float>(n); }
    const float inverse = 1.f / sum;
    constexpr float map_constant = 2.8284271247461903f;
    const float sigma2 = sigma * sigma;
    for (int i = 0; i < r; ++i) {
        rows.gather(row, B, i, r, n);
        float mean = 0.f;
        for (int col = 0; col < n; ++col) mean += weights[col] * row[col];
        mean *= inverse;
        float variance = 0.f;
        for (int col = 0; col < n; ++col) {
            const float error = row[col] - mean;
            variance += weights[col] * error * error;
        }
        row_tau[i] = map_constant * sigma2 / (std::sqrt(variance * inverse) + epsilon);
    }
    centralize(B, r, n, r, std::fabs(sigma), weights, row_tau);
}
} // namespace nss::detail
