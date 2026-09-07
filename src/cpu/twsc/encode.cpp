#include "nss/cpu_twsc.hpp"
#include "nss/cpu_api.hpp"
#include "nss/cpu_common.hpp"
#include "cpu/hwy_config.hpp"
#include "cpu/wnnm/jacobi8.hpp"
#include "cpu/finishers.hpp"

#include <algorithm>
#include <cmath>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/twsc/encode.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

int PcaProject(float* group, int m, int n, int lda, float* U, float* S, float* B, float* mean, float* work,
               int work_floats) {
    if (!group || !U || !S || !B || !mean || n <= 0 || m <= 0 || lda < m || m > kSvdMaxM || n > kSvdMaxN) {
        return -1;
    }
    const int r = std::min(m, n);
    const int svd_need = m * n * 6 + n * n * 8 + n + 256;
    if (!work || work_floats < n * n + svd_need) {
        return -1;
    }
    float* Vt = work;
    float* svd_work = Vt + n * n;
    const int svd_cap = work_floats - n * n;
    group_center_sub(group, m, n, lda, mean);
    if (svd_economy(m, n, group, lda, U, m, S, Vt, n, svd_work, svd_cap) != 0) {
        group_center_add(group, m, n, lda, mean);
        return -1;
    }
    gemm_tn_hwy(m, n, r, U, m, group, lda, B, r);
    return r;
}

void PcaReconstruct(float* group, int m, int n, int lda, const float* U, const float* B, const float* mean) {
    if (!group || !U || !B || !mean || m < 1 || n < 1 || lda < m) {
        return;
    }
    detail::finish_pca_reconstruction(group, m, n, lda, U, B, mean);
}

static void GatherRow(float* row, const float* B, int i, int r, int n) {
    const hn::ScalableTag<float> d;
    const hn::Rebind<int32_t, hn::ScalableTag<float>> di;
    const int N = static_cast<int>(hn::Lanes(d));
    const auto vr = hn::Set(di, r);
    const float* base = B + i;
    int j = 0;
    for (; j + N <= n; j += N) {
        const auto idx = hn::Mul(hn::Iota(di, static_cast<int32_t>(j)), vr);
        hn::StoreU(hn::GatherIndex(d, base, idx), d, row + j);
    }
    for (; j < n; ++j) {
        row[j] = base[j * r];
    }
}

static void ScatterRow(float* B, int i, int r, int n, const float* row) {
    const hn::ScalableTag<float> d;
    const hn::Rebind<int32_t, hn::ScalableTag<float>> di;
    const int N = static_cast<int>(hn::Lanes(d));
    const auto vr = hn::Set(di, r);
    float* base = B + i;
    int j = 0;
    for (; j + N <= n; j += N) {
        const auto idx = hn::Mul(hn::Iota(di, static_cast<int32_t>(j)), vr);
        hn::ScatterIndex(hn::LoadU(d, row + j), d, base, idx);
    }
    for (; j < n; ++j) {
        base[j * r] = row[j];
    }
}

static void ScaleRowsByW1(float* group, int m, int n, int lda, const float* row_w, bool invert) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const auto one = hn::Set(d, 1.f);
    const auto eps = hn::Set(d, 1e-12f);
    for (int j = 0; j < n; ++j) {
        float* col = group + j * lda;
        int i = 0;
        for (; i + N <= m; i += N) {
            auto w = hn::LoadU(d, row_w + i);
            if (invert) {
                w = hn::Div(hn::CopySign(one, w), hn::Max(hn::Abs(w), eps));
            }
            hn::StoreU(hn::Mul(hn::LoadU(d, col + i), w), d, col + i);
        }
        for (; i < m; ++i) {
            float w = row_w[i];
            if (invert) {
                const float a = std::fabs(w);
                w = 1.f / (a > 1e-12f ? a : 1e-12f);
                if (row_w[i] < 0.f) {
                    w = -w;
                }
            }
            col[i] *= w;
        }
    }
}

int TwscPcaSoft(float* group, int m, int n, int lda, float sigma, const float* col_sigma, float* col_w,
                const float* row_w, float* work, int work_floats) {
    if (!group || n <= 0 || m <= 0 || lda < m || m > kSvdMaxM || n > kSvdMaxN) {
        return -1;
    }
    const int need = twsc_pca_soft_work_floats(m, n);
    if (!work || work_floats < need) {
        return -1;
    }
    if (row_w) {
        ScaleRowsByW1(group, m, n, lda, row_w, false);
    }
    float* U = work;
    float* S = U + m * n;
    float* mean = S + n;
    float* B = mean + m;
    float* proj_work = B + m * n;
    const int proj_cap = work_floats - (m * n + n + m + m * n);
    const int r = PcaProject(group, m, n, lda, U, S, B, mean, proj_work, proj_cap);
    if (r < 0) {
        if (row_w) {
            ScaleRowsByW1(group, m, n, lda, row_w, true);
        }
        return -1;
    }

    struct Rows {
        void gather(float* row, const float* codes, int i, int stride, int count) const {
            GatherRow(row, codes, i, stride, count);
        }
        void scatter(float* codes, int i, int stride, int count, const float* row) const {
            ScatterRow(codes, i, stride, count, row);
        }
    };
    detail::finish_twsc_codes(r, n, sigma, col_sigma, col_w, S, B, Rows{});
    PcaReconstruct(group, m, n, lda, U, B, mean);
    if (row_w) {
        ScaleRowsByW1(group, m, n, lda, row_w, true);
    }
    return 0;
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(PcaProject);
HWY_EXPORT(PcaReconstruct);
HWY_EXPORT(TwscPcaSoft);

int pca_project(float* group, int m, int n, int lda, float* U, float* S, float* B, float* mean, float* work,
                int work_floats) {
    return HWY_DYNAMIC_DISPATCH(PcaProject)(group, m, n, lda, U, S, B, mean, work, work_floats);
}

void pca_reconstruct(float* group, int m, int n, int lda, const float* U, const float* B, const float* mean) {
    HWY_DYNAMIC_DISPATCH(PcaReconstruct)(group, m, n, lda, U, B, mean);
}

int twsc_pca_soft(float* group, int m, int n, int lda, float sigma, float* work, int work_floats,
                  const float* col_sigma, float* col_w, const float* row_w) {
    return HWY_DYNAMIC_DISPATCH(TwscPcaSoft)(group, m, n, lda, sigma, col_sigma, col_w, row_w, work, work_floats);
}

}  // namespace nss
#endif
