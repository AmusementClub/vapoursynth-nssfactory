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

// x = L^-T z (L lower triangular, row-major with stride cap): row t takes the
// later rows in descending order, so the device can run it column by column.
static void LsscBackSolve(const double* chol, int cap, const double* z, int k, double* x, float* xf) {
    for (int t = k - 1; t >= 0; --t) {
        double v = z[t];
        for (int q = k - 1; q > t; --q) v = std::fma(-chol[static_cast<std::size_t>(q) * cap + t], x[q], v);
        x[t] = v / chol[static_cast<std::size_t>(t) * cap + t];
        xf[t] = static_cast<float>(x[t]);
    }
}

int LsscOmpCode(const float* patch, const LsscOmpDictionary* dict, float sigma, float* rec, float* work) {
    const int m = dict->m;
    const int atoms = dict->atoms;
    const int cap = lssc_omp_cap(m, atoms);
    float* alpha0 = work;
    float* alpha = alpha0 + atoms;
    float* available = alpha + atoms;
    float* y = available + atoms;
    int support[kLsscOmpMaxAtoms];
    float xf[kLsscOmpMaxAtoms];
    double chol[kLsscOmpMaxAtoms * kLsscOmpMaxAtoms];  // lower triangle, row-major, stride cap
    double z[kLsscOmpMaxAtoms];
    double x[kLsscOmpMaxAtoms];
    const float* rows[kLsscOmpMaxAtoms];
    const float* cols[kLsscOmpMaxAtoms];

    double sum = 0.0;
    for (int r = 0; r < m; ++r) sum += static_cast<double>(patch[r]);
    const float mean = static_cast<float>(sum / m);
    double yy = 0.0;
    for (int r = 0; r < m; ++r) {
        y[r] = patch[r] - mean;
        yy = std::fma(static_cast<double>(y[r]), static_cast<double>(y[r]), yy);
    }
    // alpha0 = D^T y: one fma chain over the pixels per atom (Dt is pixel-major).
    const float* transpose = dict->transpose;
    LsscCombine(alpha0, nullptr, atoms, m, [&](int r) { return transpose + static_cast<std::size_t>(r) * atoms; }, y);
    std::memcpy(alpha, alpha0, static_cast<std::size_t>(atoms) * sizeof(float));
    std::fill_n(available, atoms, 1.f);
    const double c = static_cast<double>(kLsscOmpResidual) * static_cast<double>(sigma);
    const double budget = static_cast<double>(m) * c * c;
    double energy = yy;
    int k = 0;
    while (energy > budget && k < cap) {
        const int best = LsscSelect(alpha, dict->inv_norm, available, atoms);
        if (best < 0) break;
        const float* gb = dict->gram + static_cast<std::size_t>(best) * atoms;
        double* lk = chol + static_cast<std::size_t>(k) * cap;
        for (int t = 0; t < k; ++t) {
            const double* lt = chol + static_cast<std::size_t>(t) * cap;
            double v = static_cast<double>(gb[support[t]]);
            for (int q = 0; q < t; ++q) v = std::fma(-lt[q], lk[q], v);
            lk[t] = v / lt[t];
        }
        const double gbb = static_cast<double>(gb[best]);
        double pivot = gbb;
        for (int q = 0; q < k; ++q) pivot = std::fma(-lk[q], lk[q], pivot);
        if (!(pivot > static_cast<double>(kLsscOmpPivot) * gbb)) break;  // numerically dependent on the support
        lk[k] = std::sqrt(pivot);
        support[k] = best;
        available[best] = 0.f;
        rows[k] = gb;
        cols[k] = dict->dictionary + static_cast<std::size_t>(best) * dict->ldd;
        // The forward solution of the earlier rows does not change: only z[k]
        // is new, and ||r||^2 = ||y||^2 - ||z||^2.
        {
            double v = static_cast<double>(alpha0[best]);
            for (int q = 0; q < k; ++q) v = std::fma(-lk[q], z[q], v);
            z[k] = v / lk[k];
            energy = std::fma(-z[k], z[k], energy);
        }
        ++k;
        if (energy > budget && k < cap) {
            LsscBackSolve(chol, cap, z, k, x, xf);
            LsscCombine(alpha, alpha0, atoms, k, [&](int t) { return rows[t]; }, xf);
        }
    }
    if (k > 0) LsscBackSolve(chol, cap, z, k, x, xf);
    LsscCombine(rec, nullptr, m, k, [&](int t) { return cols[t]; }, xf);
    bool ok = true;
    for (int r = 0; r < m; ++r) {
        if (!std::isfinite(rec[r]) || std::fabs(rec[r]) > 8.f) ok = false;
        rec[r] += mean;
    }
    if (!ok) std::memcpy(rec, patch, static_cast<std::size_t>(m) * sizeof(float));
    return k;
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

int lssc_omp_code(const float* patch, const LsscOmpDictionary& dict, float sigma, float* rec, float* work) {
    return HWY_DYNAMIC_DISPATCH(LsscOmpCode)(patch, &dict, sigma, rec, work);
}

void lssc_omp_prepare(const float* D, int m, int atoms, int ldd, float* storage, LsscOmpDictionary* out) {
    HWY_DYNAMIC_DISPATCH(LsscOmpPrepare)(D, m, atoms, ldd, storage, out);
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

float lssc_image_weight(int block, int step, float sigma) {
    const float coverage = static_cast<float>(block * block) / static_cast<float>(step * step);
    return kLsscImageWeight * coverage / (sigma * 255.f);
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
    float* rec = code_work + lssc_omp_code_work_floats(m, atoms);
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
    for (int j = 0; j < np; ++j) {
        const float* patch = patches + static_cast<std::size_t>(j) * m;
        bool finite = true;
        for (int r = 0; r < m; ++r) finite = finite && std::isfinite(patch[r]);
        if (finite) {
            lssc_omp_code(patch, dict, sigma, rec, code_work);
        } else {
            std::memcpy(rec, patch, static_cast<std::size_t>(m) * sizeof(float));
        }
        const int bx = std::min((j % nx) * step, width - block);
        const int by = std::min((j / nx) * step, height - block);
        aggregate_add(num, den, buf_stride, bx, by, rec, block, width, height, 1.f);
    }
    image_term(num, den, buf_stride, src, sstride, width, height, lssc_image_weight(block, step, sigma));
}

}  // namespace nss
#endif
