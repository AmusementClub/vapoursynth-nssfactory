// SPDX-License-Identifier: GPL-2.0-only
#include "nss/resources.hpp"
#include "nss/cpu_api.hpp"
#include "nss/cpu_common.hpp"
#include "nss/cpu_lssc.hpp"
#include "cpu/hwy_config.hpp"
#include "cpu/wnnm/jacobi8.hpp"
#include "cpu/lssc/gemm.hpp"

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
    nss::ResourceVector<float> nrm2_v;
    nss::ResourceVector<float> scale_v;
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
    const auto vlam = hn::Set(d, lam);
    const auto one = hn::Set(d, 1.f);
    const auto z = hn::Zero(d);
    int i = 0;
    for (; i + N <= atoms; i += N) {
        const auto nrm = hn::Sqrt(hn::LoadU(d, nrm2 + i));
        const auto keep = hn::Gt(nrm, vlam);
        const auto denom = hn::IfThenElse(keep, nrm, one);
        hn::StoreU(hn::IfThenElse(keep, hn::Sub(one, hn::Div(vlam, denom)), z), d, scale + i);
    }
    for (; i < atoms; ++i) {
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

// A = A + mu G, then the group soft threshold of every atom row at
// lambda * lambda_scale[row] (lambda_scale null: 1). With `previous`, the
// same pass extrapolates: X = the thresholded A, A = X + beta (X - previous)
// (beta = 0: A = X, previous not read), previous = X.
static bool LsscUpdateGroupSoft(float* A, const float* G, int atoms, int n, float mu, float lambda,
                                const float* lambda_scale, float absolute_limit, int* active, int* active_count,
                                float* previous = nullptr, float beta = 0.f) {
    const float lam = lambda < 0.f ? 0.f : lambda;
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    float nrm2_s[512];
    float scale_s[512];
    nss::ResourceVector<float> nrm2_v;
    nss::ResourceVector<float> scale_v;
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
    const auto vlam = hn::Set(d, lam);
    const auto one = hn::Set(d, 1.f);
    const auto z = hn::Zero(d);
    int s = 0;
    for (; s + N <= atoms; s += N) {
        const auto nrm = hn::Sqrt(hn::LoadU(d, nrm2 + s));
        const auto row_lam = lambda_scale ? hn::Mul(vlam, hn::LoadU(d, lambda_scale + s)) : vlam;
        const auto keep = hn::Gt(nrm, row_lam);
        const auto denom = hn::IfThenElse(keep, nrm, one);
        hn::StoreU(hn::IfThenElse(keep, hn::Sub(one, hn::Div(row_lam, denom)), z), d, scale + s);
    }
    for (; s < atoms; ++s) {
        const float nrm = std::sqrt(nrm2[s]);
        const float row_lam = lambda_scale ? lam * lambda_scale[s] : lam;
        scale[s] = (nrm <= row_lam) ? 0.f : (1.f - row_lam / nrm);
    }
    // A zero scale makes the whole atom row exactly (signed) zero below.
    int count = 0;
    for (int i = 0; i < atoms; ++i) {
        if (scale[i] != 0.f) active[count++] = i;
    }
    *active_count = count;

    bool exploded = false;
    const auto vlim = hn::Set(d, absolute_limit);
    const auto vbeta = hn::Set(d, beta);
    for (int j = 0; j < n; ++j) {
        float* aj = A + static_cast<std::size_t>(j) * static_cast<std::size_t>(atoms);
        float* pj = previous ? previous + static_cast<std::size_t>(j) * static_cast<std::size_t>(atoms) : nullptr;
        int i = 0;
        for (; i + N <= atoms; i += N) {
            const auto scaled = hn::Mul(hn::LoadU(d, aj + i), hn::LoadU(d, scale + i));
            if (!hn::AllTrue(d, hn::And(hn::IsFinite(scaled), hn::Le(hn::Abs(scaled), vlim)))) {
                exploded = true;
            }
            if (pj && beta != 0.f) {
                hn::StoreU(hn::MulAdd(vbeta, hn::Sub(scaled, hn::LoadU(d, pj + i)), scaled), d, aj + i);
            } else {
                hn::StoreU(scaled, d, aj + i);
            }
            if (pj) hn::StoreU(scaled, d, pj + i);
        }
        for (; i < atoms; ++i) {
            const float scaled = aj[i] * scale[i];
            if (!is_finite_bits(scaled) || std::fabs(scaled) > absolute_limit) {
                exploded = true;
            }
            aj[i] = pj && beta != 0.f ? std::fma(beta, scaled - pj[i], scaled) : scaled;
            if (pj) pj[i] = scaled;
        }
    }
    return exploded;
}

// Coding of one cluster (params/lssc.hpp): `gradient` is P (atoms x m,
// column-major), `lambda_scale` the per-atom penalty factors (null: 1) and
// `lipschitz` the step bound L' of the problem P defines.
static void LsscReconstructImpl(float* patches, int m, int n, int lda, const float* dictionary, int atoms, int ldd,
                                const float* gradient, const float* lambda_scale, float lipschitz, float sigma,
                                float* work, int work_floats, bool avx2_gemm = false) {
    if (!patches || !dictionary || !gradient || m < 1 || n < 1 || atoms < 1 || lda < m || ldd < m) {
        return;
    }
    const int need = lssc_reconstruct_prepared_work_floats(m, n, atoms);
    nss::ResourceVector<float> store;
    float* buf = work;
    if (!buf || work_floats < need) {
        store.assign(static_cast<std::size_t>(need), 0.f);
        buf = store.data();
    }
    float* mean = buf;
    float* A = mean + n;
    float* R = A + atoms * n;
    float* G = R + m * n;
    float* previous = G + atoms * n;  // the last proximal iterate
    float* gemm_work = previous + atoms * n;
    const int gemm_work_floats = need - static_cast<int>(gemm_work - buf);

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
    constexpr float kALim = 1.0e4f;
    bool exploded = false;
    // Group soft-thresholding zeroes whole atom rows of A. D*A then only
    // needs the active atoms: the GEMM accumulates each output along k with
    // one FMA chain, and skipping exact zero coefficients leaves every partial
    // sum unchanged (up to the sign of an exact zero result). A holds the
    // extrapolated point Z = X + beta (X - X_prev), whose rows are zero where
    // both iterates' rows are: its active rows are the union of the two sets.
    nss::ResourceVector<int> active(static_cast<std::size_t>(atoms));
    nss::ResourceVector<unsigned char> live(static_cast<std::size_t>(2) * static_cast<std::size_t>(atoms), 0);
    unsigned char* live_now = live.data();
    unsigned char* live_previous = live_now + atoms;
    nss::ResourceVector<float> packed_dictionary;
    int active_count = atoms;
    auto product = [&]() {
        if (active_count * 4 >= atoms * 3) {
            lssc_gemm_nn(m, n, atoms, dictionary, ldd, A, atoms, R, m, gemm_work, gemm_work_floats, avx2_gemm);
            return;
        }
        if (packed_dictionary.empty()) {
            packed_dictionary.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(atoms));
        }
        float* dact = packed_dictionary.data();
        float* aact = G;  // G is dead until the following P*R product.
        for (int t = 0; t < active_count; ++t) {
            const int atom = active[t];
            std::memcpy(dact + static_cast<std::size_t>(t) * m, dictionary + static_cast<std::size_t>(atom) * ldd,
                        static_cast<std::size_t>(m) * sizeof(float));
        }
        for (int j = 0; j < n; ++j) {
            const float* aj = A + static_cast<std::size_t>(j) * atoms;
            float* bj = aact + static_cast<std::size_t>(j) * active_count;
            for (int t = 0; t < active_count; ++t) bj[t] = aj[active[t]];
        }
        lssc_gemm_nn(m, n, active_count, dact, m, aact, active_count > 0 ? active_count : 1, R, m, gemm_work,
                     gemm_work_floats, avx2_gemm);
    };
    for (int it = 0; it < kLsscCodingRounds; ++it) {
        // Z starts at zero, so the first residual is the centered input.
        // Avoid the D*0 product and the following full residual write/read.
        if (it == 0) {
            lssc_gemm_nn(atoms, n, m, gradient, atoms, patches, lda, G, atoms, gemm_work,
                         gemm_work_floats, avx2_gemm);
        } else {
            product();
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
            lssc_gemm_nn(atoms, n, m, gradient, atoms, R, m, G, atoms, gemm_work, gemm_work_floats, avx2_gemm);
        }
        // Every round but the last extrapolates in the threshold pass
        // (kLsscMomentum[0] = 0: previous is written before it is read).
        const bool last = it + 1 == kLsscCodingRounds;
        const float beta = last ? 0.f : kLsscMomentum[it];
        exploded = LsscUpdateGroupSoft(A, G, atoms, n, mu, lam, lambda_scale, kALim, active.data(), &active_count,
                                       last ? nullptr : previous, beta);
        if (exploded) {
            std::memset(A, 0, static_cast<std::size_t>(atoms) * static_cast<std::size_t>(n) * sizeof(float));
            active_count = 0;
            break;
        }
        if (last) {
            break;  // A holds the last iterate
        }
        std::memset(live_now, 0, static_cast<std::size_t>(atoms));
        for (int t = 0; t < active_count; ++t) live_now[active[t]] = 1;
        if (beta != 0.f) {
            int count = 0;
            for (int i = 0; i < atoms; ++i) {
                if (live_now[i] || live_previous[i]) active[count++] = i;
            }
            active_count = count;
        }
        std::swap(live_now, live_previous);
    }
    product();
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
    nss::ResourceVector<float> store;
    float* buf = work;
    if (!buf || work_floats < need) {
        store.assign(static_cast<std::size_t>(need), 0.f);
        buf = store.data();
    }
    // The legacy contract keeps its prepared context after the group
    // workspace; prepare it in-place before reconstruction.
    float* prep = buf + lssc_reconstruct_prepared_work_floats(m, n, atoms);
    LsscPreparedContext context;
    const int prep_need = lssc_prepare_work_floats(m, atoms);
    if (lssc_prepare_context(D, m, atoms, ldd, prep, prep_need, &context) != 0) {
        return;
    }
    LsscReconstructImpl(patches, m, n, lda, D, atoms, ldd, context.gradient, context.lambda_scale,
                        context.coding_lipschitz, sigma, buf, lssc_reconstruct_prepared_work_floats(m, n, atoms));
}

void LsscReconstructPrepared(float* patches, int m, int n, int lda, const LsscPreparedContext* context, float sigma,
                             float* work, int work_floats) {
    if (!context) {
        return;
    }
    LsscReconstructImpl(patches, m, n, lda, context->dictionary, context->atoms, context->ldd, context->gradient,
                        context->lambda_scale, context->coding_lipschitz, sigma, work, work_floats,
                        context->avx2_gemm);
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

// The coding preconditioner (params/lssc.hpp), in the order of the device
// kernels (src/cuda/lssc/kernels.cu): S = D D^T, each entry one fma chain
// over ascending atoms; a right-looking Cholesky factor of S + delta I (lower
// triangle of `factor`, row-major); per atom the forward and back solves,
// each sum one fma chain in ascending index order. Returns false when there
// is no usable factor (a trace that is not positive and finite, or a pivot
// that is not): the caller then codes with D^T, factors 1 and L.
static bool lssc_precondition(const float* D, int m, int atoms, int ldd, float lipschitz, float* factor, float* y,
                              float* gradient, float* lambda_scale, float* coding_lipschitz) {
    const std::size_t mm = static_cast<std::size_t>(m);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j <= i; ++j) {
            float sum = 0.f;
            for (int a = 0; a < atoms; ++a) {
                const float* col = D + static_cast<std::size_t>(a) * static_cast<std::size_t>(ldd);
                sum = std::fma(col[i], col[j], sum);
            }
            factor[i * mm + j] = sum;
        }
    }
    float trace = 0.f;
    for (int i = 0; i < m; ++i) {
        trace += factor[i * mm + i];
    }
    const float mean = trace / static_cast<float>(m);
    if (!(mean > 1e-12f) || !is_finite_bits(mean)) {
        return false;
    }
    const float delta = kLsscPrecondAlpha * mean;
    const float c = mean + delta;
    for (int i = 0; i < m; ++i) {
        factor[i * mm + i] += delta;
    }
    for (int k = 0; k < m; ++k) {
        const float pivot = factor[k * mm + k];
        if (!(pivot > 0.f) || !is_finite_bits(pivot)) {
            return false;
        }
        const float root = std::sqrt(pivot);
        factor[k * mm + k] = root;
        for (int i = k + 1; i < m; ++i) {
            factor[i * mm + k] /= root;
        }
        for (int j = k + 1; j < m; ++j) {
            for (int i = j; i < m; ++i) {
                factor[i * mm + j] = std::fma(-factor[i * mm + k], factor[j * mm + k], factor[i * mm + j]);
            }
        }
    }
    for (int a = 0; a < atoms; ++a) {
        const float* col = D + static_cast<std::size_t>(a) * static_cast<std::size_t>(ldd);
        for (int i = 0; i < m; ++i) {
            float sum = col[i];
            for (int p = 0; p < i; ++p) {
                sum = std::fma(-factor[i * mm + p], y[p], sum);
            }
            y[i] = sum / factor[i * mm + i];
        }
        for (int i = m - 1; i >= 0; --i) {
            float sum = y[i];
            for (int p = i + 1; p < m; ++p) {
                sum = std::fma(-factor[p * mm + i], y[p], sum);
            }
            y[i] = sum / factor[i * mm + i];
        }
        float norm2 = 0.f;
        for (int i = 0; i < m; ++i) {
            const float v = c * y[i];
            gradient[static_cast<std::size_t>(a) + static_cast<std::size_t>(i) * static_cast<std::size_t>(atoms)] = v;
            norm2 = std::fma(v, v, norm2);
        }
        lambda_scale[a] = std::sqrt(norm2);
    }
    // L' = max_e c e / (e + delta) over the eigenvalues e of S: e / (e + delta)
    // increases with e, so the largest eigenvalue (the power estimate) gives it.
    *coding_lipschitz = c * lipschitz / (lipschitz + delta);
    return true;
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
    float* gradient = pw + m;
    float* lambda_scale = gradient + static_cast<std::size_t>(atoms) * static_cast<std::size_t>(m);
    float* factor = lambda_scale + atoms;
    float coding_lipschitz = lipschitz;
    const bool preconditioned =
        lssc_precondition(D, m, atoms, ldd, lipschitz, factor, pw, gradient, lambda_scale, &coding_lipschitz);
    context->dictionary = D;
    context->transpose = transpose;
    context->gradient = preconditioned ? gradient : transpose;
    context->lambda_scale = preconditioned ? lambda_scale : nullptr;
    context->m = m;
    context->atoms = atoms;
    context->ldd = ldd;
    context->lipschitz = lipschitz;
    context->coding_lipschitz = preconditioned ? coding_lipschitz : lipschitz;
    context->avx2_gemm = false;
    return 0;
}

}  // namespace nss
#endif
