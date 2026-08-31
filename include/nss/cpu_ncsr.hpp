#pragma once

#include "nss/cpu_twsc.hpp"
#include "nss/params.hpp"

namespace nss {

inline int ncsr_filter_work_floats(int m, int n) {
    return twsc_pca_soft_work_floats(m, n);
}

inline int ncsr_denoise_work_floats(int width, int height, int block, int group) {
    const int m = block < 1 ? 1 : block * block;
    const int lda = (m + 15) & ~15;
    const int g = group < 1 ? 1 : group;
    const int plane = (width < 1 || height < 1) ? 1 : width * height;
    return plane * 4 + lda * g + ncsr_filter_work_floats(m, g) + 64;
}

// Coding-domain NSS: β = weighted mean of code columns, B ← β + soft(B−β, τ).
// col_w[j] ≥ 0; nullptr → equal weights. Rejects r or n above kSvdMaxN.
// row_tau[i] overrides tau for code row i (nullptr → scalar tau).
// Filter-group τ_i = 2√2 σ² / σ_θ(α−β) (TIP 2013 eq. 17, orthogonal PCA).
void ncsr_centralize_codes(float* B, int r, int n, int ldb, float tau, const float* col_w = nullptr,
                           const float* row_tau = nullptr);

// SVD + code-domain shrink. In-place. col_dist[j] is match distance of column j (nullptr → SSD vs col 0).
int ncsr_filter_group(float* group, int m, int n, int lda, float sigma, const float* col_dist, float* work,
                      int work_floats);

// One BM pass: match on refs, pack from srcs, filter each group, aggregate_add.
// Caller zeros num/den. work holds ncsr_filter_work_floats(block², cfg.group).
void ncsr_run_groups(const float* const* refs, const int* rstrides, const float* const* srcs, const int* sstrides,
                     int ntemp, int t0, int width, int height, const SearchConfig& cfg, float sigma, float* num,
                     float* den, float* patches, float* work);

// Spatial outer loop: regularize then denoise; last output is the shrinkage.
// work holds ncsr_denoise_work_floats floats.
void ncsr_denoise_plane(const float* src, int width, int height, int sstride, float* dst, int dstride, int block,
                        int step, int group, int bm_range, float sigma, int iters, float delta, float* work,
                        int work_floats);

}  // namespace nss
