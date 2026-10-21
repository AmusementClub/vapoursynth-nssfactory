// SPDX-License-Identifier: GPL-2.0-only
// LSSC model version 4 (include/nss/params/lssc.hpp): every grid patch is coded
// on its own by error-constrained Batch-OMP against the plane's dictionary and
// the reconstructions are averaged, with a weight of the noisy plane added.
// The arithmetic order is part of the model, shared with the CUDA kernels
// (src/cuda/lssc/kernels.cu): FP32 correlations as one fma chain each, FP64
// norms, Cholesky factor, solves and residual energy. Pinned TU (no
// contraction): every fused operation below is explicit.
#include "nss/resources.hpp"
#include "nss/avx2_policy.hpp"
#include "nss/cpu_api.hpp"
#include "nss/cpu_common.hpp"
#include "nss/cpu_lssc.hpp"
#include "cpu/hwy_config.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/lssc/code.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// out[a] = from[a] - sum_t x[t] row(t)[a] (from null: the sum alone) for
// a < count: one fma chain per a in t order. Eight vectors of a are carried
// at once so the chains overlap; the order inside each chain does not change.
template <class Row>
static void LsscCombine(float* out, const float* from, int count, int k, const Row& row, const float* x) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const auto finish = [&](int a, decltype(hn::Zero(d)) acc) {
        hn::StoreU(from ? hn::Sub(hn::LoadU(d, from + a), acc) : acc, d, out + a);
    };
    int a = 0;
    for (; a + 8 * N <= count; a += 8 * N) {
        auto a0 = hn::Zero(d), a1 = hn::Zero(d), a2 = hn::Zero(d), a3 = hn::Zero(d);
        auto a4 = hn::Zero(d), a5 = hn::Zero(d), a6 = hn::Zero(d), a7 = hn::Zero(d);
        for (int t = 0; t < k; ++t) {
            const float* r = row(t) + a;
            const auto c = hn::Set(d, x[t]);
            a0 = hn::MulAdd(c, hn::LoadU(d, r), a0);
            a1 = hn::MulAdd(c, hn::LoadU(d, r + N), a1);
            a2 = hn::MulAdd(c, hn::LoadU(d, r + 2 * N), a2);
            a3 = hn::MulAdd(c, hn::LoadU(d, r + 3 * N), a3);
            a4 = hn::MulAdd(c, hn::LoadU(d, r + 4 * N), a4);
            a5 = hn::MulAdd(c, hn::LoadU(d, r + 5 * N), a5);
            a6 = hn::MulAdd(c, hn::LoadU(d, r + 6 * N), a6);
            a7 = hn::MulAdd(c, hn::LoadU(d, r + 7 * N), a7);
        }
        finish(a, a0);
        finish(a + N, a1);
        finish(a + 2 * N, a2);
        finish(a + 3 * N, a3);
        finish(a + 4 * N, a4);
        finish(a + 5 * N, a5);
        finish(a + 6 * N, a6);
        finish(a + 7 * N, a7);
    }
    for (; a + 2 * N <= count; a += 2 * N) {
        auto a0 = hn::Zero(d), a1 = hn::Zero(d);
        for (int t = 0; t < k; ++t) {
            const float* r = row(t) + a;
            const auto c = hn::Set(d, x[t]);
            a0 = hn::MulAdd(c, hn::LoadU(d, r), a0);
            a1 = hn::MulAdd(c, hn::LoadU(d, r + N), a1);
        }
        finish(a, a0);
        finish(a + N, a1);
    }
    for (; a + N <= count; a += N) {
        auto acc = hn::Zero(d);
        for (int t = 0; t < k; ++t) acc = hn::MulAdd(hn::Set(d, x[t]), hn::LoadU(d, row(t) + a), acc);
        finish(a, acc);
    }
    for (; a < count; ++a) {
        float acc = 0.f;
        for (int t = 0; t < k; ++t) acc = std::fma(x[t], row(t)[a], acc);
        out[a] = from ? from[a] - acc : acc;
    }
}

// u = (g - sum_q l[q] u_q) * inv and alpha -= z u, per atom: one fma chain
// over q in order, then the product and the update (the device's order).
static void LsscGramSchmidt(float* u, const float* g, const float* const* prior, const float* l, int k, float inv,
                            float z, float* alpha, int atoms) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const auto vinv = hn::Set(d, inv);
    const auto vz = hn::Set(d, -z);
    int a = 0;
    for (; a + 4 * N <= atoms; a += 4 * N) {
        auto v0 = hn::LoadU(d, g + a), v1 = hn::LoadU(d, g + a + N), v2 = hn::LoadU(d, g + a + 2 * N),
             v3 = hn::LoadU(d, g + a + 3 * N);
        for (int q = 0; q < k; ++q) {
            const auto c = hn::Set(d, -l[q]);
            const float* uq = prior[q] + a;
            v0 = hn::MulAdd(c, hn::LoadU(d, uq), v0);
            v1 = hn::MulAdd(c, hn::LoadU(d, uq + N), v1);
            v2 = hn::MulAdd(c, hn::LoadU(d, uq + 2 * N), v2);
            v3 = hn::MulAdd(c, hn::LoadU(d, uq + 3 * N), v3);
        }
        v0 = hn::Mul(v0, vinv);
        v1 = hn::Mul(v1, vinv);
        v2 = hn::Mul(v2, vinv);
        v3 = hn::Mul(v3, vinv);
        hn::StoreU(v0, d, u + a);
        hn::StoreU(v1, d, u + a + N);
        hn::StoreU(v2, d, u + a + 2 * N);
        hn::StoreU(v3, d, u + a + 3 * N);
        hn::StoreU(hn::MulAdd(vz, v0, hn::LoadU(d, alpha + a)), d, alpha + a);
        hn::StoreU(hn::MulAdd(vz, v1, hn::LoadU(d, alpha + a + N)), d, alpha + a + N);
        hn::StoreU(hn::MulAdd(vz, v2, hn::LoadU(d, alpha + a + 2 * N)), d, alpha + a + 2 * N);
        hn::StoreU(hn::MulAdd(vz, v3, hn::LoadU(d, alpha + a + 3 * N)), d, alpha + a + 3 * N);
    }
    for (; a + N <= atoms; a += N) {
        auto v = hn::LoadU(d, g + a);
        for (int q = 0; q < k; ++q) v = hn::MulAdd(hn::Set(d, -l[q]), hn::LoadU(d, prior[q] + a), v);
        v = hn::Mul(v, vinv);
        hn::StoreU(v, d, u + a);
        hn::StoreU(hn::MulAdd(vz, v, hn::LoadU(d, alpha + a)), d, alpha + a);
    }
    for (; a < atoms; ++a) {
        float v = g[a];
        for (int q = 0; q < k; ++q) v = std::fma(-l[q], prior[q][a], v);
        v = v * inv;
        u[a] = v;
        alpha[a] = std::fma(-z, v, alpha[a]);
    }
}

// Unused atom of the largest |alpha| * inv_norm (available = 1, used = 0);
// the lowest index on ties, -1 when every score is 0.
static int LsscSelect(const float* alpha, const float* inv_norm, const float* available, int atoms) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const auto score = [&](int a) {
        return hn::Mul(hn::Mul(hn::Abs(hn::LoadU(d, alpha + a)), hn::LoadU(d, inv_norm + a)), hn::LoadU(d, available + a));
    };
    auto top = hn::Zero(d);
    int a = 0;
    for (; a + N <= atoms; a += N) top = hn::Max(top, score(a));
    float best_value = hn::ReduceMax(d, top);
    for (int t = a; t < atoms; ++t) best_value = std::max(best_value, std::fabs(alpha[t]) * inv_norm[t] * available[t]);
    if (!(best_value > 0.f)) return -1;
    const auto target = hn::Set(d, best_value);
    for (a = 0; a + N <= atoms; a += N) {
        const auto hit = hn::Eq(score(a), target);
        if (!hn::AllFalse(d, hit)) return a + static_cast<int>(hn::FindKnownFirstTrue(d, hit));
    }
    for (; a < atoms; ++a)
        if (std::fabs(alpha[a]) * inv_norm[a] * available[a] == best_value) return a;
    return -1;
}

// The device's warp sum: 32 lanes, lane l taking terms l, l + 32, ... in
// order from 0, then a butterfly over lane distances 16, 8, 4, 2, 1. Lane 0
// of the butterfly is the halving tree below (lane l pairs with l + half).
template <class T, class Term>
static T LsscLaneSum(int count, const Term& term) {
    T lane[32];
    for (int l = 0; l < 32; ++l) {
        T acc = 0;
        for (int t = l; t < count; t += 32) acc = term(acc, t);
        lane[l] = acc;
    }
    for (int half = 16; half > 0; half >>= 1)
        for (int l = 0; l < half; ++l) lane[l] = lane[l] + lane[l + half];
    return lane[0];
}

// x = L^-T z (L lower triangular, row-major with stride cap; inv its
// reciprocal diagonal): row t takes the later rows in descending order, so the
// device can run it column by column.
static void LsscBackSolve(const float* chol, const float* inv, int cap, const float* z, int k, float* x) {
    for (int t = k - 1; t >= 0; --t) {
        float v = z[t];
        for (int q = k - 1; q > t; --q) v = std::fma(-chol[static_cast<std::size_t>(q) * cap + t], x[q], v);
        x[t] = v * inv[t];
    }
}

// alpha_in: alpha0 = D^T y of this patch already computed (LsscAlphaBatch), or null.
static int LsscOmpCodeFrom(const float* patch, const LsscOmpDictionary* dict, float sigma, float* rec, float* work,
                           const float* alpha_in) {
    const int m = dict->m;
    const int atoms = dict->atoms;
    const int cap = lssc_omp_cap(m, atoms);
    float* alpha0 = work;
    float* alpha = alpha0 + atoms;
    float* available = alpha + atoms;
    float* y = available + atoms;
    float* ubuf = y + m;  // cap rows of atoms: u_t = G_S L^-T e_t
    int support[kLsscOmpMaxAtoms];
    float chol[kLsscOmpMaxAtoms * kLsscOmpMaxAtoms];
    float inv[kLsscOmpMaxAtoms];
    float z[kLsscOmpMaxAtoms];
    float x[kLsscOmpMaxAtoms];
    const float* urows[kLsscOmpMaxAtoms];
    const float* cols[kLsscOmpMaxAtoms];

    const float mean = LsscLaneSum<float>(m, [&](float acc, int r) { return acc + patch[r]; }) / static_cast<float>(m);
    for (int r = 0; r < m; ++r) y[r] = patch[r] - mean;
    const float yy = LsscLaneSum<float>(m, [&](float acc, int r) { return std::fma(y[r], y[r], acc); });
    // alpha0 = D^T y: one fma chain over the pixels per atom (Dt is pixel-major).
    const float* transpose = dict->transpose;
    if (alpha_in) {
        std::memcpy(alpha0, alpha_in, static_cast<std::size_t>(atoms) * sizeof(float));
    } else {
        LsscCombine(alpha0, nullptr, atoms, m, [&](int r) { return transpose + static_cast<std::size_t>(r) * atoms; }, y);
    }
    std::memcpy(alpha, alpha0, static_cast<std::size_t>(atoms) * sizeof(float));
    std::fill_n(available, atoms, 1.f);
    const float budget = lssc_omp_budget(m, sigma);
    float energy = yy;
    int k = 0;
    while (energy > budget && k < cap) {
        const int best = LsscSelect(alpha, dict->inv_norm, available, atoms);
        if (best < 0) break;
        const float* gb = dict->gram + static_cast<std::size_t>(best) * atoms;
        // Row k of the factor: L^-1 g, g_t = G[best][S_t].
        float* lk = chol + static_cast<std::size_t>(k) * cap;
        for (int t = 0; t < k; ++t) {
            const float* lt = chol + static_cast<std::size_t>(t) * cap;
            float v = gb[support[t]];
            for (int q = 0; q < t; ++q) v = std::fma(-lt[q], lk[q], v);
            lk[t] = v * inv[t];
        }
        const float gbb = gb[best];
        const float pivot = gbb - LsscLaneSum<float>(k, [&](float acc, int q) { return std::fma(lk[q], lk[q], acc); });
        if (!(pivot > kLsscOmpPivot * gbb)) break;  // numerically dependent on the support
        lk[k] = std::sqrt(pivot);
        inv[k] = 1.f / lk[k];
        support[k] = best;
        available[best] = 0.f;
        cols[k] = dict->dictionary + static_cast<std::size_t>(best) * dict->ldd;
        // The forward solution of the earlier rows does not change: only z[k]
        // is new, and ||r||^2 = ||y||^2 - ||z||^2.
        z[k] = (alpha0[best] - LsscLaneSum<float>(k, [&](float acc, int q) { return std::fma(lk[q], z[q], acc); })) * inv[k];
        energy = std::fma(-z[k], z[k], energy);
        // alpha -= z_k u_k (the residual's correlations), unless this was the last atom.
        if (energy > budget && k + 1 < cap) {
            float* uk = ubuf + static_cast<std::size_t>(k) * atoms;
            LsscGramSchmidt(uk, gb, urows, lk, k, inv[k], z[k], alpha, atoms);
            urows[k] = uk;
        }
        ++k;
    }
    if (k > 0) LsscBackSolve(chol, inv, cap, z, k, x);
    LsscCombine(rec, nullptr, m, k, [&](int t) { return cols[t]; }, x);
    bool ok = true;
    for (int r = 0; r < m; ++r) {
        if (!std::isfinite(rec[r]) || std::fabs(rec[r]) > 8.f) ok = false;
        rec[r] += mean;
    }
    if (!ok) std::memcpy(rec, patch, static_cast<std::size_t>(m) * sizeof(float));
    return k;
}

int LsscOmpCode(const float* patch, const LsscOmpDictionary* dict, float sigma, float* rec, float* work) {
    return LsscOmpCodeFrom(patch, dict, sigma, rec, work, nullptr);
}

// alpha0 = D^T y of kLsscCodeBatch patches (y: kLsscCodeBatch rows of m) with every row of D^T loaded once for all
// of them: each (patch, atom) entry is the same fma chain over the pixels in order as LsscCombine, so the result
// is bit-identical to coding the patches one by one. Per patch, D^T (64 KB at m = 64) streamed from L2 bounded
// the single-patch product; six patches share each load (2.1x on alpha0, 1.2 to 1.4x on a plane).
static void LsscAlphaBatch(const float* transpose, int atoms, int m, const float* y, float* out) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    static_assert(kLsscCodeBatch == 6, "six accumulator pairs below");
    if (atoms % (2 * N) != 0) {
        for (int i = 0; i < kLsscCodeBatch; ++i) {
            const float* yi = y + static_cast<std::size_t>(i) * m;
            LsscCombine(out + static_cast<std::size_t>(i) * atoms, nullptr, atoms, m,
                        [&](int r) { return transpose + static_cast<std::size_t>(r) * atoms; }, yi);
        }
        return;
    }
    for (int a = 0; a < atoms; a += 2 * N) {
        auto c00 = hn::Zero(d), c01 = hn::Zero(d), c10 = hn::Zero(d), c11 = hn::Zero(d);
        auto c20 = hn::Zero(d), c21 = hn::Zero(d), c30 = hn::Zero(d), c31 = hn::Zero(d);
        auto c40 = hn::Zero(d), c41 = hn::Zero(d), c50 = hn::Zero(d), c51 = hn::Zero(d);
        for (int r = 0; r < m; ++r) {
            const float* row = transpose + static_cast<std::size_t>(r) * atoms + a;
            const auto d0 = hn::LoadU(d, row), d1 = hn::LoadU(d, row + N);
            auto c = hn::Set(d, y[r]);
            c00 = hn::MulAdd(c, d0, c00);
            c01 = hn::MulAdd(c, d1, c01);
            c = hn::Set(d, y[m + r]);
            c10 = hn::MulAdd(c, d0, c10);
            c11 = hn::MulAdd(c, d1, c11);
            c = hn::Set(d, y[2 * m + r]);
            c20 = hn::MulAdd(c, d0, c20);
            c21 = hn::MulAdd(c, d1, c21);
            c = hn::Set(d, y[3 * m + r]);
            c30 = hn::MulAdd(c, d0, c30);
            c31 = hn::MulAdd(c, d1, c31);
            c = hn::Set(d, y[4 * m + r]);
            c40 = hn::MulAdd(c, d0, c40);
            c41 = hn::MulAdd(c, d1, c41);
            c = hn::Set(d, y[5 * m + r]);
            c50 = hn::MulAdd(c, d0, c50);
            c51 = hn::MulAdd(c, d1, c51);
        }
        float* o = out + a;
        hn::StoreU(c00, d, o);
        hn::StoreU(c01, d, o + N);
        hn::StoreU(c10, d, o + atoms);
        hn::StoreU(c11, d, o + atoms + N);
        hn::StoreU(c20, d, o + 2 * atoms);
        hn::StoreU(c21, d, o + 2 * atoms + N);
        hn::StoreU(c30, d, o + 3 * atoms);
        hn::StoreU(c31, d, o + 3 * atoms + N);
        hn::StoreU(c40, d, o + 4 * atoms);
        hn::StoreU(c41, d, o + 4 * atoms + N);
        hn::StoreU(c50, d, o + 5 * atoms);
        hn::StoreU(c51, d, o + 5 * atoms + N);
    }
}

// Codes count <= kLsscCodeBatch consecutive patches (rows of m) into recs, each exactly as lssc_omp_code: a patch
// with a value that is not finite is copied. A full batch shares the alpha0 product (LsscAlphaBatch).
void LsscOmpCodeBatch(const float* patches, int count, const LsscOmpDictionary* dict, float sigma, float* recs,
                      float* work) {
    const int m = dict->m, atoms = dict->atoms;
    float y[kLsscCodeBatch * kLsscMaxPatchPixels];
    float alpha0[kLsscCodeBatch * kLsscDefaultAtoms];
    bool finite[kLsscCodeBatch];
    for (int i = 0; i < count; ++i) {
        const float* patch = patches + static_cast<std::size_t>(i) * m;
        finite[i] = true;
        for (int r = 0; r < m; ++r) finite[i] = finite[i] && std::isfinite(patch[r]);
    }
    const bool shared = count == kLsscCodeBatch && m <= kLsscMaxPatchPixels && atoms <= kLsscDefaultAtoms;
    if (shared) {
        // y as LsscOmpCodeFrom centers it (its own copy stays the reference for everything after alpha0).
        for (int i = 0; i < count; ++i) {
            const float* patch = patches + static_cast<std::size_t>(i) * m;
            float* yi = y + static_cast<std::size_t>(i) * m;
            const float mean = LsscLaneSum<float>(m, [&](float acc, int r) { return acc + patch[r]; }) / static_cast<float>(m);
            for (int r = 0; r < m; ++r) yi[r] = patch[r] - mean;
        }
        LsscAlphaBatch(dict->transpose, atoms, m, y, alpha0);
    }
    for (int i = 0; i < count; ++i) {
        const float* patch = patches + static_cast<std::size_t>(i) * m;
        float* rec = recs + static_cast<std::size_t>(i) * m;
        if (finite[i]) {
            LsscOmpCodeFrom(patch, dict, sigma, rec, work, shared ? alpha0 + static_cast<std::size_t>(i) * atoms : nullptr);
        } else {
            std::memcpy(rec, patch, static_cast<std::size_t>(m) * sizeof(float));
        }
    }
}

void LsscOmpPrepare(const float* D, int m, int atoms, int ldd, float* storage, LsscOmpDictionary* out) {
    float* transpose = storage;
    float* gram = transpose + static_cast<std::size_t>(atoms) * m;
    float* inv_norm = gram + static_cast<std::size_t>(atoms) * atoms;
    for (int a = 0; a < atoms; ++a) {
        const float* da = D + static_cast<std::size_t>(a) * ldd;
        double norm = 0.0;
        for (int r = 0; r < m; ++r) {
            transpose[static_cast<std::size_t>(r) * atoms + a] = da[r];
            norm = std::fma(static_cast<double>(da[r]), static_cast<double>(da[r]), norm);
        }
        inv_norm[a] = norm > 0.0 ? static_cast<float>(1.0 / std::sqrt(norm)) : 0.f;
    }
    for (int a = 0; a < atoms; ++a) {
        const float* da = D + static_cast<std::size_t>(a) * ldd;
        for (int b = a; b < atoms; ++b) {
            const float* db = D + static_cast<std::size_t>(b) * ldd;
            float acc = 0.f;
            for (int r = 0; r < m; ++r) acc = std::fma(da[r], db[r], acc);
            gram[static_cast<std::size_t>(a) * atoms + b] = acc;
            gram[static_cast<std::size_t>(b) * atoms + a] = acc;
        }
    }
    *out = LsscOmpDictionary{D, transpose, gram, inv_norm, m, atoms, ldd};
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(LsscOmpCode);
HWY_EXPORT(LsscOmpPrepare);
HWY_EXPORT(LsscOmpCodeBatch);

int lssc_omp_code(const float* patch, const LsscOmpDictionary& dict, float sigma, float* rec, float* work) {
    return HWY_DYNAMIC_DISPATCH(LsscOmpCode)(patch, &dict, sigma, rec, work);
}

void lssc_omp_prepare(const float* D, int m, int atoms, int ldd, float* storage, LsscOmpDictionary* out) {
    HWY_DYNAMIC_DISPATCH(LsscOmpPrepare)(D, m, atoms, ldd, storage, out);
}

void lssc_omp_code_batch(const float* patches, int count, const LsscOmpDictionary& dict, float sigma, float* recs,
                         float* work) {
    HWY_DYNAMIC_DISPATCH(LsscOmpCodeBatch)(patches, count, &dict, sigma, recs, work);
}

static void image_term(float* num, float* den, int stride, const float* src, int sstride, int width, int height,
                       float weight) {
    for (int yy = 0; yy < height; ++yy) {
        float* n = num + static_cast<std::size_t>(yy) * stride;
        float* d = den + static_cast<std::size_t>(yy) * stride;
        const float* s = src + static_cast<std::size_t>(yy) * sstride;
        for (int xx = 0; xx < width; ++xx) {
            n[xx] = std::fma(weight, s[xx], n[xx]);
            d[xx] += weight;
        }
    }
}

// The plane (model version 4): grid patches, the dictionary of
// lssc_dict_init_workspace, then each patch coded alone and added with weight
// 1, and the noisy plane added with lssc_image_weight.
void lssc_denoise_plane(const float* src, int width, int height, int sstride, float* num, float* den, int buf_stride,
                        int block, int step, float sigma, float* work, int work_floats) {
    if (!src || !num || !den || width < 1 || height < 1 || sstride < width || buf_stride < width) {
        return;
    }
    for (int y = 0; y < height; ++y) {
        std::memset(num + static_cast<std::size_t>(y) * buf_stride, 0, static_cast<std::size_t>(width) * sizeof(float));
        std::memset(den + static_cast<std::size_t>(y) * buf_stride, 0, static_cast<std::size_t>(width) * sizeof(float));
    }
    if (block < 1 || step < 1 || width < block || height < block || !(sigma > 0.f) || !std::isfinite(sigma)) {
        return;
    }
    const int nx = lssc_axis_count(width, block, step);
    const int np = lssc_grid_count(width, height, block, step);
    if (np < 1) {
        return;
    }
    const int m = checked_int(static_cast<std::uint64_t>(block) * block);
    const int atoms = std::min(kLsscDefaultAtoms, np);
    const int work_need = lssc_denoise_work_floats(width, height, block, step);
    ResourceVector<float> store;
    float* buf = work;
    if (!buf || work_floats < work_need) {
        store.assign(static_cast<std::size_t>(work_need), 0.f);
        buf = store.data();
    }
    float* patches = buf;
    float* D = patches + static_cast<std::size_t>(np) * m;
    float* dict_storage = D + static_cast<std::size_t>(m) * atoms;
    float* dict_work = dict_storage + lssc_omp_dictionary_floats(m, atoms);
    const int dict_n = lssc_dict_work_floats(m, atoms, np, 1);
    float* code_work = dict_work + dict_n;
    std::memset(D, 0, static_cast<std::size_t>(m) * atoms * sizeof(float));
    for (int j = 0; j < np; ++j) {
        const int bx = std::min((j % nx) * step, width - block);
        const int by = std::min((j / nx) * step, height - block);
        pack_patch(patches + static_cast<std::size_t>(j) * m, m, src, sstride, bx, by, block, width, height);
    }
    const bool avx2_gemm = NSS_AVX2_DEFAULTS && block == 8 && step == 8;
    lssc_dict_init_workspace(D, m, atoms, m, patches, np, m, block, 1, 0x4C535343u, dict_work, dict_n, avx2_gemm);
    LsscOmpDictionary dict;
    lssc_omp_prepare(D, m, atoms, m, dict_storage, &dict);
    // Coded kLsscCodeBatch patches at a time, added in patch order.
    float recs[kLsscCodeBatch * kLsscMaxPatchPixels];
    for (int first = 0; first < np; first += kLsscCodeBatch) {
        const int count = std::min(kLsscCodeBatch, np - first);
        lssc_omp_code_batch(patches + static_cast<std::size_t>(first) * m, count, dict, sigma, recs, code_work);
        for (int i = 0; i < count; ++i) {
            const int j = first + i;
            const int bx = std::min((j % nx) * step, width - block);
            const int by = std::min((j / nx) * step, height - block);
            aggregate_add(num, den, buf_stride, bx, by, recs + static_cast<std::size_t>(i) * m, block, width, height, 1.f);
        }
    }
    image_term(num, den, buf_stride, src, sstride, width, height, lssc_image_weight(block, step, sigma));
}

}  // namespace nss
#endif
