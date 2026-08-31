#pragma once

#include "nss/cpu_api.hpp"
#include "nss/params.hpp"

namespace nss {

inline int mcwnnm_admm_work_floats(int m, int n) {
    const int mats = 4 * m * n + m * n + n + n * n + m;
    const int svd = m * n * 6 + n * n * 8 + n + 256;
    return mats + svd;
}

inline int mcwnnm_filter_work_floats(int m, int n) {
    return mcwnnm_admm_work_floats(m, n) + m;
}

// Y is m×n column-major (m = nch·p²). In-place. sigma[nch] already /255.
// W_c = min(σ)/σ_c on each channel's rows. Returns kept SV count, or -1.
int mcwnnm_admm(float* Y, int m, int n, int lda, int nch, const float* sigma, int admm_iter, float rho, float mu,
                int sv_start_k, float* work, int work_floats);

int mcwnnm_filter_group(float* Y, int m, int n, int lda, int nch, const float* sigma, int admm_iter, float rho,
                        float mu, int residual, int adaptive, float* adaptive_weight, float* work, int work_floats);

// Sum of per-plane SSD; does not modify match.cpp. Includes the reference block.
int spatial_match_nch(const float* const* refs, const int* strides, int nch, int width, int height, int bx, int by,
                      int block, int bm_range, int group, Match* out);

// refs[c * ntemp + t], strides[c]. Same predictive control flow as predictive_match.
int predictive_match_nch(const float* const* refs, const int* strides, int nch, int ntemp, int width, int height,
                         int bx, int by, int t0, const SearchConfig& cfg, Match* out);

}  // namespace nss
