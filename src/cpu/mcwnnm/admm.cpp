#include "nss/cpu_mcwnnm.hpp"
#include "nss/cpu_api.hpp"
#include "nss/cpu_common.hpp"
#include "cpu/hwy_config.hpp"
#include "cpu/wnnm/jacobi8.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/mcwnnm/admm.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

static void DualAdd(float* A, const float* X, const float* Z, int m, int n, float rho) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const auto vr = hn::Set(d, rho);
    for (int j = 0; j < n; ++j) {
        float* acol = A + j * m;
        const float* xcol = X + j * m;
        const float* zcol = Z + j * m;
        int i = 0;
        for (; i + N <= m; i += N) {
            hn::StoreU(hn::MulAdd(vr, hn::Sub(hn::LoadU(d, xcol + i), hn::LoadU(d, zcol + i)), hn::LoadU(d, acol + i)),
                       d, acol + i);
        }
        for (; i < m; ++i) {
            acol[i] += rho * (xcol[i] - zcol[i]);
        }
    }
}

static void TempXp(float* Temp, const float* X, const float* A, int m, int n, float inv_rho) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const auto vinv = hn::Set(d, inv_rho);
    for (int j = 0; j < n; ++j) {
        float* tcol = Temp + j * m;
        const float* xcol = X + j * m;
        const float* acol = A + j * m;
        int i = 0;
        for (; i + N <= m; i += N) {
            hn::StoreU(hn::MulAdd(vinv, hn::LoadU(d, acol + i), hn::LoadU(d, xcol + i)), d, tcol + i);
        }
        for (; i < m; ++i) {
            tcol[i] = xcol[i] + acol[i] * inv_rho;
        }
    }
}

static void CopyBack(float* Y, int lda, const float* Z, int m, int n) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    for (int j = 0; j < n; ++j) {
        float* ycol = Y + j * lda;
        const float* zcol = Z + j * m;
        int i = 0;
        for (; i + N <= m; i += N) {
            hn::StoreU(hn::LoadU(d, zcol + i), d, ycol + i);
        }
        for (; i < m; ++i) {
            ycol[i] = zcol[i];
        }
    }
}

int McwnnmAdmm(float* Y, int m, int n, int lda, int nch, const float* sigma, int admm_iter, float rho0, float mu,
               int sv_start_k, float* work, int work_floats) {
    if (!Y || !sigma || m < 1 || n < 1 || lda < m || nch < 1 || m % nch != 0 || m > kSvdMaxM || n > kSvdMaxN ||
        admm_iter < 1 || !(rho0 > 0.f) || !std::isfinite(rho0) || !std::isfinite(mu) || mu < 1.f) {
        return -1;
    }
    const int need = mcwnnm_admm_work_floats(m, n);
    if (!work || work_floats < need) {
        return -1;
    }
    float* X = work;
    float* Z = X + m * n;
    float* A = Z + m * n;
    float* Temp = A + m * n;
    float* U = Temp + m * n;
    float* S = U + m * n;
    float* Vt = S + n;
    float* w2 = Vt + n * n;
    float* svd_work = w2 + m;
    const int svd_cap = work_floats - (4 * m * n + m * n + n + n * n + m);

    const float smin = channel_weight_diag(w2, m, nch, sigma);
    if (!(smin > 0.f)) {
        return std::min(m, n);
    }

    std::memset(Z, 0, static_cast<std::size_t>(m * n) * sizeof(float));
    std::memset(A, 0, static_cast<std::size_t>(m * n) * sizeof(float));

    const float C = 8.f * std::sqrt(2.0f * static_cast<float>(n)) * smin * smin;
    const int start_k = std::clamp(sv_start_k, 0, std::min(m, n));
    float rho = rho0;
    int kept = 0;

    for (int it = 0; it < admm_iter; ++it) {
        admm_weighted_x(X, Y, lda, Z, A, w2, m, n, rho);
        const float inv_rho = 1.f / rho;
        TempXp(Temp, X, A, m, n, inv_rho);
        if (svd_economy(m, n, Temp, m, U, m, S, Vt, n, svd_work, svd_cap) != 0) {
            return -1;
        }
        kept = sv_shrink(S, std::min(m, n), C * (2.f / rho), start_k);
        if (kept <= 0) {
            std::memset(Z, 0, static_cast<std::size_t>(m * n) * sizeof(float));
        } else {
            if (m < n) {
                for (int t = 0; t < kept; ++t) {
                    scale_n(U + t * m, S[t], m);
                }
            } else {
                for (int j = 0; j < n; ++j) {
                    for (int t = 0; t < kept; ++t) {
                        Vt[t + j * n] *= S[t];
                    }
                }
            }
            gemm_nn_hwy(m, n, kept, U, m, Vt, n, Z, m);
        }
        DualAdd(A, X, Z, m, n, rho);
        rho = std::min(1e4f, mu * rho);
        if (!(rho > 0.f) || !std::isfinite(rho)) {
            return -1;
        }
    }

    CopyBack(Y, lda, Z, m, n);
    return kept;
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(McwnnmAdmm);

int mcwnnm_admm(float* Y, int m, int n, int lda, int nch, const float* sigma, int admm_iter, float rho, float mu,
                int sv_start_k, float* work, int work_floats) {
    return HWY_DYNAMIC_DISPATCH(McwnnmAdmm)(Y, m, n, lda, nch, sigma, admm_iter, rho, mu, sv_start_k, work,
                                            work_floats);
}

}  // namespace nss
#endif
