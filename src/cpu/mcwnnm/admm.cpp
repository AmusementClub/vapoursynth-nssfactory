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

// MCWNNM only needs the shrunk reconstruction, not the SVD factors. For the
// common tall 8-column case, diagonalize A^T A and apply the resulting spectral
// map directly to A. This avoids forming and replaying the tall QR basis.
static int GramShrink8(const float* input, int m, float constant, int start_k, float* output) {
    constexpr int kN = 8;
    HWY_ALIGN float gram[kN * kN];
    HWY_ALIGN float eig_u[kN * kN];
    HWY_ALIGN float eig_s[kN];
    HWY_ALIGN float eig_vt[kN * kN];
    HWY_ALIGN float eig_v[kN * kN];
    HWY_ALIGN float singular[kN];
    HWY_ALIGN float spectral_factor[kN];
    HWY_ALIGN float transform[kN * kN];

    gemm_tn_hwy(m, kN, kN, input, m, input, m, gram, kN);
    jacobi_svd_8(gram, kN, eig_u, kN, eig_s, eig_vt, kN, eig_v);
    for (int k = 0; k < kN; ++k) {
        if (!is_finite_bits(eig_s[k]) || eig_s[k] < 0.f) {
            return -1;
        }
        singular[k] = std::sqrt(eig_s[k]);
    }
    const float uncertainty = 2e-5f * std::max({eig_s[0], constant, 1e-20f});
    for (int k = std::clamp(start_k, 0, kN); k < kN; ++k) {
        if (std::fabs(eig_s[k] - constant) <= uncertainty) {
            return -1;
        }
    }
    const int kept = sv_shrink(singular, kN, constant, start_k);
    for (int k = 0; k < kept; ++k) {
        const float original = std::sqrt(eig_s[k]);
        spectral_factor[k] = original > 1e-20f ? singular[k] / original : 0.f;
    }
    for (int col = 0; col < kN; ++col) {
        for (int row = 0; row <= col; ++row) {
            float value = 0.f;
            for (int k = 0; k < kept; ++k) {
                value += eig_vt[k + row * kN] * spectral_factor[k] * eig_vt[k + col * kN];
            }
            if (!is_finite_bits(value)) {
                return -1;
            }
            transform[row + col * kN] = value;
            transform[col + row * kN] = value;
        }
    }
    gemm_nn_hwy(m, kN, kN, input, m, transform, kN, output, m);
    return kept;
}

static int McwnnmAdmmGram8(float* Y, int m, int lda, int nch, const float* sigma, int admm_iter, float rho0,
                           float mu, float* work) {
    constexpr int n = 8;
    float* X = work;
    float* Z = X + m * n;
    float* A = Z + m * n;
    float* Temp = A + m * n;
    float* U = Temp + m * n;
    float* S = U + m * n;
    float* Vt = S + n;
    float* w2 = Vt + n * n;

    const float smin = channel_weight_diag(w2, m, nch, sigma);
    if (!(smin > 0.f)) {
        return n;
    }
    std::memset(Z, 0, static_cast<std::size_t>(m * n) * sizeof(float));
    std::memset(A, 0, static_cast<std::size_t>(m * n) * sizeof(float));

    const float C = 8.f * std::sqrt(2.0f * static_cast<float>(n)) * smin * smin;
    float rho = rho0;
    int kept = 0;
    for (int it = 0; it < admm_iter; ++it) {
        admm_weighted_x(X, Y, lda, Z, A, w2, m, n, rho);
        TempXp(Temp, X, A, m, n, 1.f / rho);
        kept = GramShrink8(Temp, m, C * (2.f / rho), 0, Z);
        if (kept < 0) {
            return -1;
        }
        DualAdd(A, X, Z, m, n, rho);
        rho = std::min(1e4f, mu * rho);
        if (!(rho > 0.f) || !is_finite_bits(rho)) {
            return -1;
        }
    }
    CopyBack(Y, lda, Z, m, n);
    return kept;
}

int McwnnmAdmm(float* Y, int m, int n, int lda, int nch, const float* sigma, int admm_iter, float rho0, float mu,
               int sv_start_k, float* work, int work_floats) {
    if (!Y || !sigma || m < 1 || n < 1 || lda < m || nch < 1 || m % nch != 0 || m > kSvdMaxM || n > kSvdMaxN ||
        admm_iter < 1 || !(rho0 > 0.f) || !is_finite_bits(rho0) || !is_finite_bits(mu) || mu < 1.f) {
        return -1;
    }
    const int need = mcwnnm_admm_work_floats(m, n);
    if (!work || work_floats < need) {
        return -1;
    }
    // A^T A squares the condition number. Restrict the direct spectral path
    // to the well-conditioned tall shapes observed for legal 3-channel block
    // sizes 8 and 9. The original QR body below remains the fail-closed path.
    if (n == 8 && m >= 192 && sv_start_k == 0) {
        const int gram_kept = McwnnmAdmmGram8(Y, m, lda, nch, sigma, admm_iter, rho0, mu, work);
        if (gram_kept >= 0) {
            return gram_kept;
        }
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
        if (!(rho > 0.f) || !is_finite_bits(rho)) {
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
