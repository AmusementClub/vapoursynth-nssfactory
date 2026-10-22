// SPDX-License-Identifier: GPL-2.0-only
// DA3D model version 1 (include/nss/params/da3d.hpp). Adapted from the IPOL
// reference implementation by Nicola Pierazzo (BSD-2-Clause, see NOTICE): the
// tile split, weight map pyramid and per-patch steps follow it; the patch sums
// are 32-lane sums and the DFT is this file's own (two real transforms packed
// in one complex 64x64 transform). Pinned TU (no contraction): every fused
// operation below is explicit.
#include "nss/cpu_da3d.hpp"
#include "nss/params/da3d.hpp"
#include "cpu/hwy_config.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/da3d/da3d.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

#ifndef NSS_DA3D_ONCE
#define NSS_DA3D_ONCE
namespace nss {
namespace da3d_detail {

constexpr int kS = kDa3dPatch;
constexpr int kR = kDa3dRadius;
constexpr int kArea = kS * kS;
constexpr int kAfter = kS - kR - 1;  // padding after a tile

inline std::size_t align16(std::size_t n) { return (n + 15) & ~std::size_t{15}; }

inline int symmetric(int pos, int size) {
    if (pos < 0) pos = -pos - 1;
    if (pos >= 2 * size) pos %= 2 * size;
    if (pos >= size) pos = 2 * size - 1 - pos;
    return pos;
}

inline int next_pow2(int n) {
    --n;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

inline int number_of_bits(int n) {
    int bits = 0;
    while (n) {
        ++bits;
        n >>= 1;
    }
    return bits;
}

// The reference's weight map: level 0 holds the weight of every core pixel
// (padded with +inf to a power of two), level l the minimum of 2x2 squares
// of level l - 1.
struct WeightMap {
    static constexpr int kMaxLevels = 32;
    int levels = 0, width = 0, height = 0;
    int rows[kMaxLevels]{}, cols[kMaxLevels]{};
    float* data[kMaxLevels]{};

    static std::size_t floats(int h, int w) {
        std::size_t total = 0;
        const int lv = std::max(number_of_bits(h - 1) + 1, number_of_bits(w - 1) + 1);
        int rr = next_pow2(h), cr = next_pow2(w);
        for (int l = 0; l < lv; ++l) {
            total += align16(static_cast<std::size_t>(rr) * cr);
            rr = ((rr + 3) >> 2) << 1;
            cr = ((cr + 3) >> 2) << 1;
        }
        return total;
    }

    void init(int h, int w, float* storage) {
        levels = std::max(number_of_bits(h - 1) + 1, number_of_bits(w - 1) + 1);
        width = w;
        height = h;
        int rr = next_pow2(h), cr = next_pow2(w);
        int vr = h, vc = w;
        const float inf = std::numeric_limits<float>::infinity();
        for (int l = 0; l < levels; ++l) {
            rows[l] = rr;
            cols[l] = cr;
            data[l] = storage;
            storage += align16(static_cast<std::size_t>(rr) * cr);
            for (int r = 0; r < rr; ++r)
                for (int c = 0; c < cr; ++c) data[l][r * cr + c] = (r < vr && c < vc) ? 0.f : inf;
            vr = (vr + 1) >> 1;
            vc = (vc + 1) >> 1;
            rr = ((rr + 3) >> 2) << 1;
            cr = ((cr + 3) >> 2) << 1;
        }
    }

    float at(int l, int r, int c) const { return data[l][r * cols[l] + c]; }
    float minimum() const { return at(levels - 1, 0, 0); }

    void find_minimum(int* out_r, int* out_c) const {
        int r = 0, c = 0;
        for (int l = levels - 2; l >= 0; --l) {
            r <<= 1;
            c <<= 1;
            int br = r, bc = c;
            float best = at(l, r, c);
            const int cand[3][2] = {{r + 1, c}, {r, c + 1}, {r + 1, c + 1}};
            for (const auto& q : cand) {
                const float v = at(l, q[0], q[1]);
                if (v < best) {
                    best = v;
                    br = q[0];
                    bc = q[1];
                }
            }
            r = br;
            c = bc;
        }
        *out_r = r;
        *out_c = c;
    }
};

}  // namespace da3d_detail
}  // namespace nss
#endif  // NSS_DA3D_ONCE

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;
using namespace da3d_detail;

using DF = hn::CappedTag<float, 16>;
using VF = hn::Vec<DF>;

struct alignas(64) Tables {
    float twiddle_re[64], twiddle_im[64];  // e^{-2 pi i t / 64}
    int rev4[64];                          // base-4 digit reversal (3 digits)
    int negrow[64];                        // the row of -f in digit-reversed order
    float rr[kArea], cc[kArea];            // row - r, col - r
    float rr2[kArea], rc[kArea], cc2[kArea];
    float dist_reg[kArea], dist_k[kArea];  // d^2 / (2 sigma_sr^2), d^2 / (2 sigma_s^2)
    Tables() {
        const double pi = 3.14159265358979323846;
        for (int t = 0; t < 64; ++t) {
            twiddle_re[t] = static_cast<float>(std::cos(2.0 * pi * t / 64.0));
            twiddle_im[t] = static_cast<float>(-std::sin(2.0 * pi * t / 64.0));
        }
        for (int i = 0; i < 64; ++i) rev4[i] = (i & 3) << 4 | (i & 12) | (i >> 4);
        for (int i = 0; i < 64; ++i) negrow[i] = rev4[(64 - rev4[i]) & 63];
        const float sigma_s2 = kDa3dSigmaS * kDa3dSigmaS;
        const float sigma_sr2 = sigma_s2 * kDa3dRegressionSpatialScale;
        for (int row = 0; row < kS; ++row) {
            for (int col = 0; col < kS; ++col) {
                const int i = row * kS + col;
                const int dr = row - kR, dc = col - kR;
                rr[i] = static_cast<float>(dr);
                cc[i] = static_cast<float>(dc);
                rr2[i] = static_cast<float>(dr * dr);
                rc[i] = static_cast<float>(dr * dc);
                cc2[i] = static_cast<float>(dc * dc);
                dist_reg[i] = static_cast<float>(dr * dr + dc * dc) / (2 * sigma_sr2);
                dist_k[i] = static_cast<float>(dr * dr + dc * dc) / (2 * sigma_s2);
            }
        }
    }
};

static const Tables& tables() {
    static const Tables t;
    return t;
}

// K 32-lane sums at once (params/da3d.hpp): term(acc, i) adds the terms of
// elements i .. i + N - 1 to acc[0 .. K).
template <int K, class Term>
static HWY_NOINLINE void LaneSums(float* out, const Term& term) {
    const DF d;
    const int N = static_cast<int>(hn::Lanes(d));
    const int per = 32 / N;
    VF acc[K][32];
    for (int k = 0; k < K; ++k)
        for (int j = 0; j < per; ++j) acc[k][j] = hn::Zero(d);
    for (int i = 0; i < kArea; i += 32) {
        for (int j = 0; j < per; ++j) {
            VF a[K];
            for (int k = 0; k < K; ++k) a[k] = acc[k][j];
            term(a, i + j * N);
            for (int k = 0; k < K; ++k) acc[k][j] = a[k];
        }
    }
    for (int k = 0; k < K; ++k) {
        HWY_ALIGN float lane[32];
        for (int j = 0; j < per; ++j) hn::Store(acc[k][j], d, lane + j * N);
        for (int half = 16; half > 0; half >>= 1)
            for (int l = 0; l < half; ++l) lane[l] = lane[l] + lane[l + half];
        out[k] = lane[0];
    }
}

static HWY_INLINE VF FastExp(DF d, VF x) {
    const hn::RebindToSigned<DF> di;
    const auto t = hn::Mul(hn::Set(d, 12102203.f), x);
    const auto keep = hn::Gt(t, hn::Set(d, -1065353216.f));
    auto bits = hn::Add(hn::ConvertTo(di, hn::IfThenElse(keep, t, hn::Zero(d))), hn::Set(di, 1065353216));
    bits = hn::IfThenElseZero(hn::RebindMask(di, keep), hn::Max(bits, hn::Zero(di)));
    return hn::BitCast(d, bits);
}

// (xr, xi) times the twiddle (wr, wi): (xr wr - xi wi, xr wi + xi wr).
static HWY_INLINE void Twiddle(VF& xr, VF& xi, VF wr, VF wi) {
    const VF r = hn::MulSub(xr, wr, hn::Mul(xi, wi));
    xi = hn::MulAdd(xr, wi, hn::Mul(xi, wr));
    xr = r;
}

// Radix-4 decimation in frequency along the rows of a 64x64 complex array
// (all 64 columns at once): natural order in, base-4 digit-reversed rows out.
// Sign -1 (forward) or +1 (inverse).
static HWY_NOINLINE void Dif4(float* re, float* im, bool inverse) {
    const DF d;
    const int N = static_cast<int>(hn::Lanes(d));
    const Tables& tb = tables();
    const float sign = inverse ? -1.f : 1.f;
    for (int q = 16; q >= 1; q >>= 2) {
        const int s = 16 / q;
        for (int g = 0; g < 64; g += 4 * q) {
            for (int j = 0; j < q; ++j) {
                float* r0 = re + (g + j) * 64;
                float* i0 = im + (g + j) * 64;
                float* r1 = r0 + q * 64;
                float* i1 = i0 + q * 64;
                float* r2 = r1 + q * 64;
                float* i2 = i1 + q * 64;
                float* r3 = r2 + q * 64;
                float* i3 = i2 + q * 64;
                const int t = j * s;
                const auto w1r = hn::Set(d, tb.twiddle_re[t]), w1i = hn::Set(d, sign * tb.twiddle_im[t]);
                const auto w2r = hn::Set(d, tb.twiddle_re[2 * t]), w2i = hn::Set(d, sign * tb.twiddle_im[2 * t]);
                const auto w3r = hn::Set(d, tb.twiddle_re[3 * t]), w3i = hn::Set(d, sign * tb.twiddle_im[3 * t]);
                for (int c = 0; c < 64; c += N) {
                    const auto x0r = hn::Load(d, r0 + c), x0i = hn::Load(d, i0 + c);
                    const auto x1r = hn::Load(d, r1 + c), x1i = hn::Load(d, i1 + c);
                    const auto x2r = hn::Load(d, r2 + c), x2i = hn::Load(d, i2 + c);
                    const auto x3r = hn::Load(d, r3 + c), x3i = hn::Load(d, i3 + c);
                    const auto ar = hn::Add(x0r, x2r), ai = hn::Add(x0i, x2i);
                    const auto br = hn::Sub(x0r, x2r), bi = hn::Sub(x0i, x2i);
                    const auto cr = hn::Add(x1r, x3r), ci = hn::Add(x1i, x3i);
                    const auto dr = hn::Sub(x1r, x3r), di = hn::Sub(x1i, x3i);
                    hn::Store(hn::Add(ar, cr), d, r0 + c);
                    hn::Store(hn::Add(ai, ci), d, i0 + c);
                    // b - i d and b + i d (forward); swapped for the inverse.
                    auto mr = hn::Add(br, di), mi = hn::Sub(bi, dr);
                    auto pr = hn::Sub(br, di), pi = hn::Add(bi, dr);
                    if (inverse) {
                        std::swap(mr, pr);
                        std::swap(mi, pi);
                    }
                    auto er = hn::Sub(ar, cr), ei = hn::Sub(ai, ci);
                    if (t != 0) {
                        Twiddle(mr, mi, w1r, w1i);
                        Twiddle(er, ei, w2r, w2i);
                        Twiddle(pr, pi, w3r, w3i);
                    }
                    hn::Store(mr, d, r1 + c);
                    hn::Store(mi, d, i1 + c);
                    hn::Store(er, d, r2 + c);
                    hn::Store(ei, d, i2 + c);
                    hn::Store(pr, d, r3 + c);
                    hn::Store(pi, d, i3 + c);
                }
            }
        }
    }
}

// Radix-4 decimation in time: base-4 digit-reversed rows in, natural order out.
static HWY_NOINLINE void Dit4(float* re, float* im, bool inverse) {
    const DF d;
    const int N = static_cast<int>(hn::Lanes(d));
    const Tables& tb = tables();
    const float sign = inverse ? -1.f : 1.f;
    for (int q = 1; q <= 16; q <<= 2) {
        const int s = 16 / q;
        for (int g = 0; g < 64; g += 4 * q) {
            for (int j = 0; j < q; ++j) {
                float* r0 = re + (g + j) * 64;
                float* i0 = im + (g + j) * 64;
                float* r1 = r0 + q * 64;
                float* i1 = i0 + q * 64;
                float* r2 = r1 + q * 64;
                float* i2 = i1 + q * 64;
                float* r3 = r2 + q * 64;
                float* i3 = i2 + q * 64;
                const int t = j * s;
                const auto w1r = hn::Set(d, tb.twiddle_re[t]), w1i = hn::Set(d, sign * tb.twiddle_im[t]);
                const auto w2r = hn::Set(d, tb.twiddle_re[2 * t]), w2i = hn::Set(d, sign * tb.twiddle_im[2 * t]);
                const auto w3r = hn::Set(d, tb.twiddle_re[3 * t]), w3i = hn::Set(d, sign * tb.twiddle_im[3 * t]);
                for (int c = 0; c < 64; c += N) {
                    auto x0r = hn::Load(d, r0 + c), x0i = hn::Load(d, i0 + c);
                    auto x1r = hn::Load(d, r1 + c), x1i = hn::Load(d, i1 + c);
                    auto x2r = hn::Load(d, r2 + c), x2i = hn::Load(d, i2 + c);
                    auto x3r = hn::Load(d, r3 + c), x3i = hn::Load(d, i3 + c);
                    if (t != 0) {
                        Twiddle(x1r, x1i, w1r, w1i);
                        Twiddle(x2r, x2i, w2r, w2i);
                        Twiddle(x3r, x3i, w3r, w3i);
                    }
                    const auto ar = hn::Add(x0r, x2r), ai = hn::Add(x0i, x2i);
                    const auto br = hn::Sub(x0r, x2r), bi = hn::Sub(x0i, x2i);
                    const auto cr = hn::Add(x1r, x3r), ci = hn::Add(x1i, x3i);
                    const auto dr = hn::Sub(x1r, x3r), di = hn::Sub(x1i, x3i);
                    auto mr = hn::Add(br, di), mi = hn::Sub(bi, dr);  // b - i d
                    auto pr = hn::Sub(br, di), pi = hn::Add(bi, dr);  // b + i d
                    if (inverse) {
                        std::swap(mr, pr);
                        std::swap(mi, pi);
                    }
                    hn::Store(hn::Add(ar, cr), d, r0 + c);
                    hn::Store(hn::Add(ai, ci), d, i0 + c);
                    hn::Store(mr, d, r1 + c);
                    hn::Store(mi, d, i1 + c);
                    hn::Store(hn::Sub(ar, cr), d, r2 + c);
                    hn::Store(hn::Sub(ai, ci), d, i2 + c);
                    hn::Store(pr, d, r3 + c);
                    hn::Store(pi, d, i3 + c);
                }
            }
        }
    }
}

// dst[c][f] = src[perm[f]][c] (perm null: the identity), in 4x4 blocks.
static HWY_NOINLINE void Transpose(const float* src, float* dst, const int* perm) {
#if HWY_TARGET == HWY_SCALAR
    for (int f = 0; f < 64; ++f) {
        const float* s = src + (perm ? perm[f] : f) * 64;
        for (int c = 0; c < 64; ++c) dst[c * 64 + f] = s[c];
    }
#else
    const hn::FixedTag<float, 4> d4;
    for (int f = 0; f < 64; f += 4) {
        const float* s0 = src + (perm ? perm[f] : f) * 64;
        const float* s1 = src + (perm ? perm[f + 1] : f + 1) * 64;
        const float* s2 = src + (perm ? perm[f + 2] : f + 2) * 64;
        const float* s3 = src + (perm ? perm[f + 3] : f + 3) * 64;
        for (int c = 0; c < 64; c += 4) {
            const auto a0 = hn::Load(d4, s0 + c), a1 = hn::Load(d4, s1 + c);
            const auto a2 = hn::Load(d4, s2 + c), a3 = hn::Load(d4, s3 + c);
            const auto t0 = hn::InterleaveLower(d4, a0, a1), t1 = hn::InterleaveLower(d4, a2, a3);
            const auto t2 = hn::InterleaveUpper(d4, a0, a1), t3 = hn::InterleaveUpper(d4, a2, a3);
            hn::Store(hn::ConcatLowerLower(d4, t1, t0), d4, dst + (c + 0) * 64 + f);
            hn::Store(hn::ConcatUpperUpper(d4, t1, t0), d4, dst + (c + 1) * 64 + f);
            hn::Store(hn::ConcatLowerLower(d4, t3, t2), d4, dst + (c + 2) * 64 + f);
            hn::Store(hn::ConcatUpperUpper(d4, t3, t2), d4, dst + (c + 3) * 64 + f);
        }
    }
#endif
}

// The forward transform of (re, im) (destroyed) into (out_re, out_im), stored
// [rev4 fc][fr]: row p holds the frequency rev4(p) along the columns.
static HWY_NOINLINE void Forward2(float* re, float* im, float* out_re, float* out_im) {
    const Tables& tb = tables();
    Dif4(re, im, false);
    Transpose(re, out_re, tb.rev4);
    Transpose(im, out_im, tb.rev4);
    Dif4(out_re, out_im, false);
}

// The inverse of that layout from (re, im) (destroyed) into (out_re, out_im),
// unnormalized and stored [rev4 row][col].
static HWY_NOINLINE void Inverse2(float* re, float* im, float* out_re, float* out_im) {
    Dit4(re, im, true);
    Transpose(re, out_re, nullptr);
    Transpose(im, out_im, nullptr);
    Dif4(out_re, out_im, true);
}

// Natural layouts for the exported test entry: x[row][col] <-> F[fc][fr].
void Da3dDft64x64(float* re, float* im, bool inverse, float* scratch) {
    const Tables& tb = tables();
    float* sre = scratch;
    float* sim = scratch + kArea;
    if (!inverse) {
        Forward2(re, im, sre, sim);
    } else {
        for (int p = 0; p < 64; ++p) {
            std::memcpy(sre + p * 64, re + tb.rev4[p] * 64, 64 * sizeof(float));
            std::memcpy(sim + p * 64, im + tb.rev4[p] * 64, 64 * sizeof(float));
        }
        Inverse2(sre, sim, re, im);
        std::memcpy(sre, re, kArea * sizeof(float));
        std::memcpy(sim, im, kArea * sizeof(float));
    }
    for (int f = 0; f < 64; ++f) {
        std::memcpy(re + f * 64, sre + tb.rev4[f] * 64, 64 * sizeof(float));
        std::memcpy(im + f * 64, sim + tb.rev4[f] * 64, 64 * sizeof(float));
    }
}

struct Patch {
    // Each array holds kArea floats.
    float* y[3];
    float* g[3];
    float* plane[3];
    float* k_reg;
    float* k;
    float* tre;  // transform work
    float* tim;
    float* zre[3];
    float* zim[3];
    float* wre[2];
    float* wim[2];
    static constexpr int kArrays = 3 * 3 + 2 + 2 + 3 * 2 + 2 * 2;
    static std::size_t floats() { return static_cast<std::size_t>(kArea) * kArrays; }
    void carve(float* p) {
        auto take = [&p]() {
            float* r = p;
            p += kArea;
            return r;
        };
        for (int c = 0; c < 3; ++c) y[c] = take();
        for (int c = 0; c < 3; ++c) g[c] = take();
        for (int c = 0; c < 3; ++c) plane[c] = take();
        k_reg = take();
        k = take();
        tre = take();
        tim = take();
        for (int c = 0; c < 3; ++c) zre[c] = take();
        for (int c = 0; c < 3; ++c) zim[c] = take();
        for (int c = 0; c < 2; ++c) wre[c] = take();
        for (int c = 0; c < 2; ++c) wim[c] = take();
    }
};

// k = fastexp(-(sum_c (g_c - g_c(p))^2 inv_range + dist)).
static HWY_NOINLINE void BilateralWeight(float* const* g, int channels, float inv_range, const float* dist, float* k) {
    const DF d;
    const int N = static_cast<int>(hn::Lanes(d));
    float center[3];
    for (int c = 0; c < channels; ++c) center[c] = g[c][kR * kS + kR];
    const auto vinv = hn::Set(d, inv_range);
    for (int i = 0; i < kArea; i += N) {
        auto e = hn::Sub(hn::Load(d, g[0] + i), hn::Set(d, center[0]));
        auto x = hn::Mul(e, e);
        for (int c = 1; c < channels; ++c) {
            e = hn::Sub(hn::Load(d, g[c] + i), hn::Set(d, center[c]));
            x = hn::MulAdd(e, e, x);
        }
        x = hn::MulAdd(x, vinv, hn::Load(d, dist + i));
        hn::Store(FastExp(d, hn::Neg(x)), d, k + i);
    }
}

// Y = (Z(f) + conj Z(-f)) / 2, G = (Z(f) - conj Z(-f)) / 2i of the packed
// transform Z = DFT(y_m + i g_m) (rows digit-reversed), and Y' = K Y with
// K = fastexp(coef / |G|^2), coef = -gamma_f sigma_f^2, off DC (K = 1 there).
// Y' goes to (outr, outi), or with kAddI, i Y' is added to them.
template <bool kAddI>
static HWY_NOINLINE void ShrinkChannel(const float* zre, const float* zim, float* outr, float* outi, float coef) {
    const DF d;
    const int N = static_cast<int>(hn::Lanes(d));
    const Tables& tb = tables();
    const auto half = hn::Set(d, 0.5f);
    const auto vcoef = hn::Set(d, coef);
    HWY_ALIGN float ext_re[80], ext_im[80];
    const auto block = [&](int i, int b0, bool dc) {
        const auto zr = hn::Load(d, zre + i), zi = hn::Load(d, zim + i);
        const auto qr = hn::Reverse(d, hn::LoadU(d, ext_re + 64 - b0 - N + 1));
        const auto qi = hn::Reverse(d, hn::LoadU(d, ext_im + 64 - b0 - N + 1));
        const auto yr = hn::Mul(half, hn::Add(zr, qr));
        const auto yi = hn::Mul(half, hn::Sub(zi, qi));
        const auto gr = hn::Mul(half, hn::Add(zi, qi));
        const auto gi = hn::Mul(half, hn::Sub(qr, zr));
        auto K = FastExp(d, hn::Div(vcoef, hn::MulAdd(gr, gr, hn::Mul(gi, gi))));
        if (dc) K = hn::InsertLane(K, 0, 1.f);
        const auto ry = hn::Mul(K, yr), iy = hn::Mul(K, yi);
        if (kAddI) {
            hn::Store(hn::Sub(hn::Load(d, outr + i), iy), d, outr + i);
            hn::Store(hn::Add(hn::Load(d, outi + i), ry), d, outi + i);
        } else {
            hn::Store(ry, d, outr + i);
            hn::Store(iy, d, outi + i);
        }
    };
    for (int a0 = 0; a0 < 64; ++a0) {
        const int na = tb.negrow[a0];
        std::memcpy(ext_re, zre + na * 64, 64 * sizeof(float));
        std::memcpy(ext_im, zim + na * 64, 64 * sizeof(float));
        ext_re[64] = ext_re[0];
        ext_im[64] = ext_im[0];
        int b0 = 0;
        if (a0 == 0) {
            block(0, 0, true);
            b0 = N;
        }
        for (; b0 < 64; b0 += N) block(a0 * 64 + b0, b0, false);
    }
}

struct TileContext {
    int channels;
    float sigma2;
    float* tile_noisy[3];
    float* tile_guide[3];
    float* tile_num[3];
    float* tile_den;
    int tile_stride;
    Patch patch;
    WeightMap wm;
    Da3dStats stats;
};

static HWY_NOINLINE void ProcessPatch(TileContext& cx, int pr, int pc) {
    const DF d;
    const int N = static_cast<int>(hn::Lanes(d));
    const Tables& tb = tables();
    const int C = cx.channels;
    Patch& P = cx.patch;
    const int ts = cx.tile_stride;

    for (int c = 0; c < C; ++c) {
        for (int row = 0; row < kS; ++row) {
            std::memcpy(P.y[c] + row * kS, cx.tile_noisy[c] + static_cast<std::ptrdiff_t>(pr + row) * ts + pc,
                        kS * sizeof(float));
            std::memcpy(P.g[c] + row * kS, cx.tile_guide[c] + static_cast<std::ptrdiff_t>(pr + row) * ts + pc,
                        kS * sizeof(float));
        }
    }

    // Regression plane through g(p), weighted by k_reg.
    const float gamma_r_sigma2 = kDa3dGammaR * cx.sigma2;
    BilateralWeight(P.g, C, 1.f / (gamma_r_sigma2 * kDa3dRegressionRangeScale), tb.dist_reg, P.k_reg);
    const float* kr = P.k_reg;
    float abc[3];
    LaneSums<3>(abc, [&](VF* acc, int i) {
        const auto kv = hn::Load(d, kr + i);
        acc[0] = hn::MulAdd(hn::Load(d, tb.rr2 + i), kv, acc[0]);
        acc[1] = hn::MulAdd(hn::Load(d, tb.rc + i), kv, acc[1]);
        acc[2] = hn::MulAdd(hn::Load(d, tb.cc2 + i), kv, acc[2]);
    });
    const float a = abc[0], b = abc[1], cq = abc[2];
    const float det = a * cq - b * b;
    float slope_r[3]{}, slope_c[3]{};
    if (!(std::fabs(det) < kDa3dRegressionEpsilon)) {
        for (int c = 0; c < C; ++c) {
            const auto central = hn::Set(d, P.g[c][kR * kS + kR]);
            const float* yc = P.y[c];
            float de[2];
            LaneSums<2>(de, [&](VF* acc, int i) {
                const auto kv = hn::Load(d, kr + i);
                const auto v = hn::Sub(hn::Load(d, yc + i), central);
                acc[0] = hn::MulAdd(hn::Mul(hn::Load(d, tb.rr + i), v), kv, acc[0]);
                acc[1] = hn::MulAdd(hn::Mul(hn::Load(d, tb.cc + i), v), kv, acc[1]);
            });
            slope_r[c] = (cq * de[0] - b * de[1]) / det;
            slope_c[c] = (a * de[1] - b * de[0]) / det;
        }
    }
    for (int c = 0; c < C; ++c) {
        const auto sr = hn::Set(d, slope_r[c]), sc = hn::Set(d, slope_c[c]);
        for (int i = 0; i < kArea; i += N) {
            const auto p = hn::MulAdd(sc, hn::Load(d, tb.cc + i), hn::Mul(sr, hn::Load(d, tb.rr + i)));
            hn::Store(p, d, P.plane[c] + i);
            hn::Store(hn::Sub(hn::Load(d, P.y[c] + i), p), d, P.y[c] + i);
            hn::Store(hn::Sub(hn::Load(d, P.g[c] + i), p), d, P.g[c] + i);
        }
    }

    BilateralWeight(P.g, C, 1.f / gamma_r_sigma2, tb.dist_k, P.k);
    const float* k = P.k;
    float mass_energy[2];
    LaneSums<2>(mass_energy, [&](VF* acc, int i) {
        const auto kv = hn::Load(d, k + i);
        acc[0] = hn::Add(acc[0], kv);
        acc[1] = hn::MulAdd(kv, kv, acc[1]);
    });
    const float mass = mass_energy[0];
    ++cx.stats.patches;

    if (mass < kDa3dGuideOnlyMass) {
        ++cx.stats.guide_only;
        for (int row = 0; row < kS; ++row) {
            float* den = cx.tile_den + static_cast<std::ptrdiff_t>(pr + row) * ts + pc;
            for (int col = 0; col < kS; col += N) {
                const int i = row * kS + col;
                const auto kv = hn::Load(d, k + i);
                const auto kk = hn::Mul(kv, kv);
                for (int c = 0; c < C; ++c) {
                    float* num = cx.tile_num[c] + static_cast<std::ptrdiff_t>(pr + row) * ts + pc + col;
                    const auto v = hn::Add(hn::Load(d, P.g[c] + i), hn::Load(d, P.plane[c] + i));
                    hn::StoreU(hn::MulAdd(v, kk, hn::LoadU(d, num)), d, num);
                }
                hn::StoreU(hn::Add(hn::LoadU(d, den + col), kk), d, den + col);
                hn::Store(kk, d, P.k + i);
            }
        }
        return;
    }

    // y_m = k y + (1 - k) avg packed with g_m as y_m + i g_m.
    const auto one = hn::Set(d, 1.f);
    float avg[3];
    for (int c = 0; c < C; ++c) {
        const float* yc = P.y[c];
        const float* gc = P.g[c];
        float sums[2];
        LaneSums<2>(sums, [&](VF* acc, int i) {
            const auto kv = hn::Load(d, k + i);
            acc[0] = hn::MulAdd(kv, hn::Load(d, yc + i), acc[0]);
            acc[1] = hn::MulAdd(kv, hn::Load(d, gc + i), acc[1]);
        });
        avg[c] = sums[0] / mass;
        const auto ya = hn::Set(d, avg[c]), ga = hn::Set(d, sums[1] / mass);
        for (int i = 0; i < kArea; i += N) {
            const auto kv = hn::Load(d, k + i);
            const auto rest = hn::Sub(one, kv);
            hn::Store(hn::MulAdd(kv, hn::Load(d, yc + i), hn::Mul(rest, ya)), d, P.tre + i);
            hn::Store(hn::MulAdd(kv, hn::Load(d, gc + i), hn::Mul(rest, ga)), d, P.tim + i);
        }
        Forward2(P.tre, P.tim, P.zre[c], P.zim[c]);
    }
    const float sigma_f2 = mass_energy[1] * cx.sigma2;

    const float coef = -kDa3dGammaF * sigma_f2;
    ShrinkChannel<false>(P.zre[0], P.zim[0], P.wre[0], P.wim[0], coef);
    if (C == 3) {
        ShrinkChannel<true>(P.zre[1], P.zim[1], P.wre[0], P.wim[0], coef);
        ShrinkChannel<false>(P.zre[2], P.zim[2], P.wre[1], P.wim[1], coef);
    }
    // y'_0 (+ i y'_1) into (tre, tim); y'_2 into wre[0].
    Inverse2(P.wre[0], P.wim[0], P.tre, P.tim);
    if (C == 3) Inverse2(P.wre[1], P.wim[1], P.wre[0], P.wim[0]);
    const float* yout[3] = {P.tre, P.tim, P.wre[0]};

    // Add ((y' - (1 - k) avg) + plane k) k and k^2.
    const auto inv_n = hn::Set(d, 1.f / static_cast<float>(kArea));
    for (int row = 0; row < kS; ++row) {
        float* den = cx.tile_den + static_cast<std::ptrdiff_t>(pr + row) * ts + pc;
        const int src_row = tb.rev4[row] * kS;
        for (int col = 0; col < kS; col += N) {
            const int i = row * kS + col;
            const auto kv = hn::Load(d, k + i);
            const auto rest = hn::Sub(one, kv);
            for (int c = 0; c < C; ++c) {
                float* num = cx.tile_num[c] + static_cast<std::ptrdiff_t>(pr + row) * ts + pc + col;
                auto v = hn::Sub(hn::Mul(hn::Load(d, yout[c] + src_row + col), inv_n), hn::Mul(rest, hn::Set(d, avg[c])));
                v = hn::MulAdd(hn::Load(d, P.plane[c] + i), kv, v);
                hn::StoreU(hn::MulAdd(v, kv, hn::LoadU(d, num)), d, num);
            }
            const auto kk = hn::Mul(kv, kv);
            hn::StoreU(hn::Add(hn::LoadU(d, den + col), kk), d, den + col);
            hn::Store(kk, d, P.k + i);
        }
    }
}

// Adds k (now k^2) to the weight map at core (pr - r, pc - r) and refreshes
// the pyramid over the touched squares.
static HWY_NOINLINE void IncreaseWeights(WeightMap& wm, const float* kk, int row0, int col0) {
    const int first_r = std::max(0, row0), last_r = std::min(wm.height, row0 + kS) - 1;
    const int first_c = std::max(0, col0), last_c = std::min(wm.width, col0 + kS) - 1;
    for (int r = first_r; r <= last_r; ++r) {
        float* dst = wm.data[0] + r * wm.cols[0];
        const float* src = kk + (r - row0) * kS - col0;
        for (int c = first_c; c <= last_c; ++c) dst[c] = dst[c] + src[c];
    }
    for (int l = 1; l < wm.levels; ++l) {
        const float* lo = wm.data[l - 1];
        const int lc = wm.cols[l - 1];
        for (int r = first_r >> l; r <= last_r >> l; ++r) {
            float* dst = wm.data[l] + r * wm.cols[l];
            for (int c = first_c >> l; c <= last_c >> l; ++c) {
                const float* q = lo + (2 * r) * lc + 2 * c;
                dst[c] = std::min({q[0], q[1], q[lc], q[lc + 1]});
            }
        }
    }
}

void Da3dDenoise(const Da3dImage& img, float* work, std::size_t work_floats, Da3dStats* stats) {
    const int C = img.channels;
    const int W = img.width, H = img.height;
    if ((C != 1 && C != 3) || W < 1 || H < 1 || !(img.sigma > 0.f))
        throw std::invalid_argument("nss: DA3D kernel arguments are invalid");
    if (work_floats < da3d_work_floats(W, H, C)) throw std::invalid_argument("nss: DA3D work buffer too small");
    const std::size_t plane = static_cast<std::size_t>(W) * H;
    const int tile_w = da3d_tile_max(W) + kS - 1;
    const int tile_h = da3d_tile_max(H) + kS - 1;
    const std::size_t tile = static_cast<std::size_t>(tile_w) * tile_h;

    float* p = work;
    auto take = [&p](std::size_t n) {
        float* r = p;
        p += align16(n);
        return r;
    };
    float* num[3]{};
    for (int c = 0; c < C; ++c) num[c] = take(plane);
    float* den = take(plane);
    TileContext cx{};
    cx.channels = C;
    cx.sigma2 = img.sigma * img.sigma;
    cx.tile_stride = tile_w;
    for (int c = 0; c < C; ++c) {
        cx.tile_noisy[c] = take(tile);
        cx.tile_guide[c] = take(tile);
        cx.tile_num[c] = take(tile);
    }
    cx.tile_den = take(tile);
    float* wm_storage = take(WeightMap::floats(da3d_tile_max(H), da3d_tile_max(W)));
    cx.patch.carve(take(Patch::floats()));
    for (int c = 0; c < C; ++c) std::fill(num[c], num[c] + plane, 0.f);
    std::fill(den, den + plane, 0.f);

    const float rs3 = std::sqrt(3.f), rs2 = std::sqrt(2.f), rs6 = std::sqrt(6.f);
    const int rows_t = da3d_tile_count(H), cols_t = da3d_tile_count(W);
    for (int tr = 0; tr < rows_t; ++tr) {
        const int r0 = da3d_tile_start(H, rows_t, tr), r1 = da3d_tile_start(H, rows_t, tr + 1);
        for (int tc = 0; tc < cols_t; ++tc) {
            const int c0 = da3d_tile_start(W, cols_t, tc), c1 = da3d_tile_start(W, cols_t, tc + 1);
            const int ph = r1 - r0 + kS - 1, pw = c1 - c0 + kS - 1;
            for (int row = 0; row < ph; ++row) {
                const int sy = symmetric(r0 - kR + row, H);
                for (int col = 0; col < pw; ++col) {
                    const int sx = symmetric(c0 - kR + col, W);
                    const std::size_t o = static_cast<std::size_t>(row) * tile_w + col;
                    float nv[3], gv[3];
                    for (int c = 0; c < C; ++c) {
                        nv[c] = img.noisy[c][sy * img.noisy_stride[c] + sx];
                        gv[c] = img.guide[c][sy * img.guide_stride[c] + sx];
                    }
                    if (img.rgb) {
                        for (float* v : {nv, gv}) {
                            const float r = v[0], g = v[1], b = v[2];
                            v[0] = (r + g + b) / rs3;
                            v[1] = (r - b) / rs2;
                            v[2] = (r - 2 * g + b) / rs6;
                        }
                    }
                    for (int c = 0; c < C; ++c) {
                        cx.tile_noisy[c][o] = nv[c];
                        cx.tile_guide[c][o] = gv[c];
                        cx.tile_num[c][o] = 0.f;
                    }
                    cx.tile_den[o] = 0.f;
                }
            }
            cx.wm.init(r1 - r0, c1 - c0, wm_storage);
            while (cx.wm.minimum() < kDa3dThreshold) {
                int pr, pc;
                cx.wm.find_minimum(&pr, &pc);
                ProcessPatch(cx, pr, pc);
                IncreaseWeights(cx.wm, cx.patch.k, pr - kR, pc - kR);
            }
            for (int row = std::max(0, r0 - kR); row < std::min(H, r1 + kAfter); ++row) {
                const int trow = row - (r0 - kR);
                for (int col = std::max(0, c0 - kR); col < std::min(W, c1 + kAfter); ++col) {
                    const std::size_t o = static_cast<std::size_t>(trow) * tile_w + (col - (c0 - kR));
                    const std::size_t q = static_cast<std::size_t>(row) * W + col;
                    for (int c = 0; c < C; ++c) num[c][q] = num[c][q] + cx.tile_num[c][o];
                    den[q] = den[q] + cx.tile_den[o];
                }
            }
        }
    }

    for (int row = 0; row < H; ++row) {
        for (int col = 0; col < W; ++col) {
            const std::size_t q = static_cast<std::size_t>(row) * W + col;
            float v[3];
            for (int c = 0; c < C; ++c) v[c] = num[c][q] / den[q];
            if (img.rgb) {
                const float y = v[0], u = v[1], w = v[2];
                v[0] = (rs2 * y + rs3 * u + w) / rs6;
                v[1] = (y - rs2 * w) / rs3;
                v[2] = (rs2 * y - rs3 * u + w) / rs6;
            }
            for (int c = 0; c < C; ++c) img.out[c][row * img.out_stride[c] + col] = v[c];
        }
    }
    if (stats) {
        stats->patches += cx.stats.patches;
        stats->guide_only += cx.stats.guide_only;
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(Da3dDenoise);
HWY_EXPORT(Da3dDft64x64);

std::size_t da3d_work_floats(int width, int height, int channels) {
    using namespace da3d_detail;
    const std::size_t plane = static_cast<std::size_t>(width) * height;
    const std::size_t tile = static_cast<std::size_t>(da3d_tile_max(width) + kS - 1) * (da3d_tile_max(height) + kS - 1);
    const std::size_t patch = static_cast<std::size_t>(kArea) * (3 * 3 + 2 + 2 + 3 * 2 + 2 * 2);
    return (channels + 1) * align16(plane) + (3 * channels + 1) * align16(tile) +
           align16(WeightMap::floats(da3d_tile_max(height), da3d_tile_max(width))) + align16(patch) + 16;
}

void da3d_denoise(const Da3dImage& img, float* work, std::size_t work_floats, Da3dStats* stats) {
    HWY_DYNAMIC_DISPATCH(Da3dDenoise)(img, work, work_floats, stats);
}

void da3d_dft64x64(float* re, float* im, bool inverse, float* scratch) {
    HWY_DYNAMIC_DISPATCH(Da3dDft64x64)(re, im, inverse, scratch);
}
}  // namespace nss
#endif
