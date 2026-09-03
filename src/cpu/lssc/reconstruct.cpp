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

static bool LsscUpdateGroupSoft(float* A, const float* G, int atoms, int n, float mu, float lambda,
                                float absolute_limit) {
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
        scale_v.resize(static_cast<std::size_t>(atoms));
        nrm2 = nrm2_v.data();
        scale = scale_v.data();
    } else {
        std::memset(nrm2_s, 0, static_cast<std::size_t>(atoms) * sizeof(float));
    }

    const auto vmu = hn::Set(d, mu);
    for (int j = 0; j < n; ++j) {
        float* aj = A + static_cast<std::size_t>(j) * static_cast<std::size_t>(atoms);
        const float* gj = G + static_cast<std::size_t>(j) * static_cast<std::size_t>(atoms);
        int i = 0;
        for (; i + N <= atoms; i += N) {
            const auto updated = hn::MulAdd(vmu, hn::LoadU(d, gj + i), hn::LoadU(d, aj + i));
            hn::StoreU(updated, d, aj + i);
            hn::StoreU(hn::MulAdd(updated, updated, hn::LoadU(d, nrm2 + i)), d, nrm2 + i);
        }
        for (; i < atoms; ++i) {
            aj[i] += mu * gj[i];
            nrm2[i] += aj[i] * aj[i];
        }
    }
    for (int i = 0; i < atoms; ++i) {
        const float nrm = std::sqrt(nrm2[i]);
        scale[i] = (nrm <= lam) ? 0.f : (1.f - lam / nrm);
    }

    bool exploded = false;
    const auto vlim = hn::Set(d, absolute_limit);
    for (int j = 0; j < n; ++j) {
        float* aj = A + static_cast<std::size_t>(j) * static_cast<std::size_t>(atoms);
        int i = 0;
        for (; i + N <= atoms; i += N) {
            const auto scaled = hn::Mul(hn::LoadU(d, aj + i), hn::LoadU(d, scale + i));
            hn::StoreU(scaled, d, aj + i);
            if (!hn::AllTrue(d, hn::And(hn::IsFinite(scaled), hn::Le(hn::Abs(scaled), vlim)))) {
                exploded = true;
            }
        }
        for (; i < atoms; ++i) {
            aj[i] *= scale[i];
            if (!is_finite_bits(aj[i]) || std::fabs(aj[i]) > absolute_limit) {
                exploded = true;
            }
        }
    }
    return exploded;
}

static void LsscReconstructImpl(float* patches, int m, int n, int lda, const float* dictionary, int atoms, int ldd,
                                const float* transpose, float lipschitz, float sigma, float* work, int work_floats) {
    if (!patches || !dictionary || !transpose || m < 1 || n < 1 || atoms < 1 || lda < m || ldd < m) {
        return;
    }
    const int need = lssc_reconstruct_prepared_work_floats(m, n, atoms);
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
    if (!(lipschitz > 1e-6f) || !is_finite_bits(lipschitz)) {
        lipschitz = 1.f;
    }
    const float mu = 0.9f / lipschitz;
    const float lam_obj = std::fabs(sigma) * std::sqrt(static_cast<float>(n));
    const float lam = mu * lam_obj;
    constexpr int kIters = 16;
    constexpr float kALim = 1.0e4f;
    bool exploded = false;
    for (int it = 0; it < kIters; ++it) {
        gemm_nn_hwy(m, n, atoms, dictionary, ldd, A, atoms, R, m);
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
        gemm_nn_hwy(atoms, n, m, transpose, atoms, R, m, G, atoms);
        exploded = LsscUpdateGroupSoft(A, G, atoms, n, mu, lam, kALim);
        if (exploded) {
            std::memset(A, 0, static_cast<std::size_t>(atoms) * static_cast<std::size_t>(n) * sizeof(float));
            break;
        }
    }
    gemm_nn_hwy(m, n, atoms, dictionary, ldd, A, atoms, R, m);
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
            float value = rj[i] + mean[j];
            if (!is_finite_bits(value) || std::fabs(rj[i]) > 8.f) {
                value = mean[j];
            }
            yj[i] = value;
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
    float* transpose = buf + lssc_reconstruct_prepared_work_floats(m, n, atoms);
    // The legacy contract keeps its transpose and power vectors after the
    // group workspace; prepare the context in-place before reconstruction.
    LsscPreparedContext context;
    const int prep_need = lssc_prepare_work_floats(m, atoms);
    float* prep = transpose;
    if (lssc_prepare_context(D, m, atoms, ldd, prep, prep_need, &context) != 0) {
        return;
    }
    LsscReconstructImpl(patches, m, n, lda, D, atoms, ldd, context.transpose, context.lipschitz, sigma, buf,
                        lssc_reconstruct_prepared_work_floats(m, n, atoms));
}

void LsscReconstructPrepared(float* patches, int m, int n, int lda, const LsscPreparedContext* context, float sigma,
                             float* work, int work_floats) {
    if (!context) {
        return;
    }
    LsscReconstructImpl(patches, m, n, lda, context->dictionary, context->atoms, context->ldd, context->transpose,
                        context->lipschitz, sigma, work, work_floats);
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(LsscGroupSoft);
HWY_EXPORT(LsscReconstruct);
HWY_EXPORT(LsscReconstructPrepared);

void lssc_group_soft(float* A, int atoms, int n, int lda_a, float lambda) {
    HWY_DYNAMIC_DISPATCH(LsscGroupSoft)(A, atoms, n, lda_a, lambda);
}

void lssc_reconstruct(float* patches, int m, int n, int lda, const float* D, int atoms, int ldd, float sigma,
                      float* work, int work_floats) {
    HWY_DYNAMIC_DISPATCH(LsscReconstruct)(patches, m, n, lda, D, atoms, ldd, sigma, work, work_floats);
}

void lssc_reconstruct_prepared(float* patches, int m, int n, int lda, const LsscPreparedContext* context, float sigma,
                               float* work, int work_floats) {
    HWY_DYNAMIC_DISPATCH(LsscReconstructPrepared)(patches, m, n, lda, context, sigma, work, work_floats);
}

int lssc_prepare_context(const float* D, int m, int atoms, int ldd, float* work, int work_floats,
                         LsscPreparedContext* context) {
    if (!D || !work || !context || m < 1 || atoms < 1 || ldd < m ||
        work_floats < lssc_prepare_work_floats(m, atoms)) {
        return -1;
    }
    float* transpose = work;
    float* pv = transpose + static_cast<std::size_t>(atoms) * static_cast<std::size_t>(m);
    float* pw = pv + atoms;
    for (int a = 0; a < atoms; ++a) {
        const float* col = D + static_cast<std::size_t>(a) * static_cast<std::size_t>(ldd);
        for (int i = 0; i < m; ++i) {
            transpose[static_cast<std::size_t>(a) + static_cast<std::size_t>(i) * static_cast<std::size_t>(atoms)] =
                col[i];
        }
    }
    const float inva = 1.f / std::sqrt(static_cast<float>(atoms));
    for (int a = 0; a < atoms; ++a) {
        pv[a] = inva;
    }
    float lipschitz = 1.f;
    for (int pit = 0; pit < 8; ++pit) {
        gemm_nn_hwy(m, 1, atoms, D, ldd, pv, atoms, pw, m);
        float w2 = 0.f;
        for (int i = 0; i < m; ++i) {
            w2 += pw[i] * pw[i];
        }
        lipschitz = w2;
        gemm_tn_hwy(m, 1, atoms, D, ldd, pw, m, pv, atoms);
        float v2 = 0.f;
        for (int a = 0; a < atoms; ++a) {
            v2 += pv[a] * pv[a];
        }
        if (!(v2 > 1e-20f) || !is_finite_bits(v2)) {
            lipschitz = 1.f;
            break;
        }
        const float inv = 1.f / std::sqrt(v2);
        for (int a = 0; a < atoms; ++a) {
            pv[a] *= inv;
        }
    }
    if (!(lipschitz > 1e-6f) || !is_finite_bits(lipschitz)) {
        lipschitz = 1.f;
    }
    context->dictionary = D;
    context->transpose = transpose;
    context->m = m;
    context->atoms = atoms;
    context->ldd = ldd;
    context->lipschitz = lipschitz;
    return 0;
}

}  // namespace nss
#endif
