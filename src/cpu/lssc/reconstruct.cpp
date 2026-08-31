#include "nss/cpu_api.hpp"
#include "nss/cpu_common.hpp"
#include "nss/cpu_lssc.hpp"
#include "cpu/hwy_config.hpp"
#include "cpu/wnnm/jacobi8.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/lssc/reconstruct.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

void LsscGroupSoft(float* A, int atoms, int n, int lda_a, float lambda) {
    if (!A || atoms < 1 || n < 1 || lda_a < atoms) {
        return;
    }
    const float lam = lambda < 0.f ? 0.f : lambda;
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    float nrm2_s[512];
    float scale_s[512];
    std::vector<float> nrm2_v;
    std::vector<float> scale_v;
    float* nrm2 = nrm2_s;
    float* scale = scale_s;
    if (atoms > 512) {
        nrm2_v.assign(static_cast<std::size_t>(atoms), 0.f);
        scale_v.assign(static_cast<std::size_t>(atoms), 0.f);
        nrm2 = nrm2_v.data();
        scale = scale_v.data();
    } else {
        std::memset(nrm2_s, 0, static_cast<std::size_t>(atoms) * sizeof(float));
    }
    for (int j = 0; j < n; ++j) {
        const float* col = A + j * lda_a;
        int i = 0;
        for (; i + N <= atoms; i += N) {
            const auto v = hn::LoadU(d, col + i);
            hn::StoreU(hn::MulAdd(v, v, hn::LoadU(d, nrm2 + i)), d, nrm2 + i);
        }
        for (; i < atoms; ++i) {
            nrm2[i] += col[i] * col[i];
        }
    }
    for (int i = 0; i < atoms; ++i) {
        const float nrm = std::sqrt(nrm2[i]);
        scale[i] = (nrm <= lam) ? 0.f : (1.f - lam / nrm);
    }
    for (int j = 0; j < n; ++j) {
        float* col = A + j * lda_a;
        int i = 0;
        for (; i + N <= atoms; i += N) {
            hn::StoreU(hn::Mul(hn::LoadU(d, col + i), hn::LoadU(d, scale + i)), d, col + i);
        }
        for (; i < atoms; ++i) {
            col[i] *= scale[i];
        }
    }
}

void LsscReconstruct(float* patches, int m, int n, int lda, const float* D, int atoms, int ldd, float sigma,
                     float* work, int work_floats) {
    if (!patches || !D || m < 1 || n < 1 || atoms < 1 || lda < m || ldd < m) {
        return;
    }
    const int need = lssc_reconstruct_work_floats(m, n, atoms);
    std::vector<float> store;
    float* buf = work;
    if (!buf || work_floats < need) {
        store.assign(static_cast<std::size_t>(need), 0.f);
        buf = store.data();
    }
    float* mean = buf;
    float* A = mean + n;
    float* R = A + atoms * n;
    float* G = R + m * n;
    float* DT = G + atoms * n;

    const float invm = 1.f / static_cast<float>(m);
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    for (int j = 0; j < n; ++j) {
        float* col = patches + static_cast<std::size_t>(j) * static_cast<std::size_t>(lda);
        auto acc = hn::Zero(d);
        int i = 0;
        for (; i + N <= m; i += N) {
            acc = hn::Add(acc, hn::LoadU(d, col + i));
        }
        float s = hn::ReduceSum(d, acc);
        for (; i < m; ++i) {
            s += col[i];
        }
        mean[j] = s * invm;
        const auto vmu = hn::Set(d, mean[j]);
        i = 0;
        for (; i + N <= m; i += N) {
            hn::StoreU(hn::Sub(hn::LoadU(d, col + i), vmu), d, col + i);
        }
        for (; i < m; ++i) {
            col[i] -= mean[j];
        }
    }

    std::memset(A, 0, static_cast<std::size_t>(atoms) * static_cast<std::size_t>(n) * sizeof(float));
    for (int a = 0; a < atoms; ++a) {
        const float* da = D + static_cast<std::size_t>(a) * static_cast<std::size_t>(ldd);
        for (int i = 0; i < m; ++i) {
            DT[static_cast<std::size_t>(a) + static_cast<std::size_t>(i) * static_cast<std::size_t>(atoms)] = da[i];
        }
    }

    // ISTA needs μ < 1/||D||₂². Fixed μ=0.5 diverges when overcomplete D has ||D||₂² ≫ 2.
    std::vector<float> pv(static_cast<std::size_t>(atoms), 0.f);
    std::vector<float> pw(static_cast<std::size_t>(m), 0.f);
    const float inva = 1.f / std::sqrt(static_cast<float>(atoms));
    for (int k = 0; k < atoms; ++k) {
        pv[static_cast<std::size_t>(k)] = inva;
    }
    float lip = 1.f;
    for (int pit = 0; pit < 8; ++pit) {
        gemm_nn_hwy(m, 1, atoms, D, ldd, pv.data(), atoms, pw.data(), m);
        float w2 = 0.f;
        for (int i = 0; i < m; ++i) {
            w2 += pw[static_cast<std::size_t>(i)] * pw[static_cast<std::size_t>(i)];
        }
        lip = w2;
        gemm_tn_hwy(m, 1, atoms, D, ldd, pw.data(), m, pv.data(), atoms);
        float v2 = 0.f;
        for (int k = 0; k < atoms; ++k) {
            v2 += pv[static_cast<std::size_t>(k)] * pv[static_cast<std::size_t>(k)];
        }
        if (!(v2 > 1e-20f) || !std::isfinite(v2)) {
            lip = 1.f;
            break;
        }
        const float inv = 1.f / std::sqrt(v2);
        for (int k = 0; k < atoms; ++k) {
            pv[static_cast<std::size_t>(k)] *= inv;
        }
    }
    if (!(lip > 1e-6f) || !std::isfinite(lip)) {
        lip = 1.f;
    }
    const float mu = 0.9f / lip;
    const float lam_obj = std::fabs(sigma) * std::sqrt(static_cast<float>(n));
    const float lam = mu * lam_obj;
    constexpr int kIters = 16;
    constexpr float kALim = 1.0e4f;
    bool exploded = false;
    for (int it = 0; it < kIters; ++it) {
        gemm_nn_hwy(m, n, atoms, D, ldd, A, atoms, R, m);
        for (int j = 0; j < n; ++j) {
            const float* yj = patches + static_cast<std::size_t>(j) * static_cast<std::size_t>(lda);
            float* rj = R + static_cast<std::size_t>(j) * static_cast<std::size_t>(m);
            int i = 0;
            for (; i + N <= m; i += N) {
                hn::StoreU(hn::Sub(hn::LoadU(d, yj + i), hn::LoadU(d, rj + i)), d, rj + i);
            }
            for (; i < m; ++i) {
                rj[i] = yj[i] - rj[i];
            }
        }
        gemm_nn_hwy(atoms, n, m, DT, atoms, R, m, G, atoms);
        const auto vmu = hn::Set(d, mu);
        for (int j = 0; j < n; ++j) {
            float* aj = A + static_cast<std::size_t>(j) * static_cast<std::size_t>(atoms);
            const float* gj = G + static_cast<std::size_t>(j) * static_cast<std::size_t>(atoms);
            int i = 0;
            for (; i + N <= atoms; i += N) {
                hn::StoreU(hn::MulAdd(vmu, hn::LoadU(d, gj + i), hn::LoadU(d, aj + i)), d, aj + i);
            }
            for (; i < atoms; ++i) {
                aj[i] += mu * gj[i];
            }
        }
        LsscGroupSoft(A, atoms, n, atoms, lam);
        for (int t = 0; t < atoms * n; ++t) {
            const float a = A[static_cast<std::size_t>(t)];
            if (!std::isfinite(a) || std::fabs(a) > kALim) {
                exploded = true;
                break;
            }
        }
        if (exploded) {
            std::memset(A, 0, static_cast<std::size_t>(atoms) * static_cast<std::size_t>(n) * sizeof(float));
            break;
        }
    }
    gemm_nn_hwy(m, n, atoms, D, ldd, A, atoms, R, m);
    const auto vlim = hn::Set(d, 8.f);
    for (int j = 0; j < n; ++j) {
        float* yj = patches + static_cast<std::size_t>(j) * static_cast<std::size_t>(lda);
        const float* rj = R + static_cast<std::size_t>(j) * static_cast<std::size_t>(m);
        const auto vmu = hn::Set(d, mean[j]);
        int i = 0;
        for (; i + N <= m; i += N) {
            const auto rec = hn::Add(hn::LoadU(d, rj + i), vmu);
            const auto ok = hn::And(hn::IsFinite(rec), hn::Le(hn::Abs(hn::Sub(rec, vmu)), vlim));
            hn::StoreU(hn::IfThenElse(ok, rec, vmu), d, yj + i);
        }
        for (; i < m; ++i) {
            float s = rj[i] + mean[j];
            if (!std::isfinite(s) || std::fabs(rj[i]) > 8.f) {
                s = mean[j];
            }
            yj[i] = s;
        }
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(LsscGroupSoft);
HWY_EXPORT(LsscReconstruct);

void lssc_group_soft(float* A, int atoms, int n, int lda_a, float lambda) {
    HWY_DYNAMIC_DISPATCH(LsscGroupSoft)(A, atoms, n, lda_a, lambda);
}

void lssc_reconstruct(float* patches, int m, int n, int lda, const float* D, int atoms, int ldd, float sigma,
                      float* work, int work_floats) {
    HWY_DYNAMIC_DISPATCH(LsscReconstruct)(patches, m, n, lda, D, atoms, ldd, sigma, work, work_floats);
}

}  // namespace nss
#endif
