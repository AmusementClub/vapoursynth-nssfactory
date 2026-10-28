// SPDX-License-Identifier: GPL-2.0-only
#pragma once
// Resolved PNLM arguments, constants and the arithmetic shared by every backend.
//
// PNLM: patchwise non-local means after Davy and Ehret, "GPU acceleration of
// NL-means, BM3D and VBM3D", J. Real-Time Image Processing 18 (2021), section
// 2.1 (Buades, Coll and Morel's NL-means as improved in their IPOL 2011
// paper: the n most similar patches, the 2 sigma^2 distance offset, the flat
// patch rule, the whole patch aggregated), model version 1. Each plane is
// filtered on its own; s = sigma / 255 is the noise standard deviation in the
// plane's units, p the patch side (block_size, 4, 6 or 8) and q = p / 2.
//
// References: the p x p patches of frame n at x = 0, q, 2q, ... and the last
// one at width - p (the same along y; pnlm_grid). For each reference, the
// candidates are the patches within +-a along x and y (inside the plane) in
// the distinct frames max(0, n - d) .. min(last, n + d), the reference itself
// excepted; the n - 1 of least key (pnlm_key: distance, then frame, row,
// column) follow the reference. The distance D is the sum of squared
// differences over the patch, in a fixed order: each q x q quarter sums its
// columns top to bottom with fmaf, the columns are added as pnlm_columns, the
// quarters as (top left + top right) + (bottom left + bottom right).
// Weight of a match: pnlm_exp(-max(D / p^2 - 2 s^2, 0) / h^2) with h = 2 s
// (the reference 1). The estimate of the patch is sum w P / sum w, the sums in
// match order (fmaf); with r the variance of all the k p^2 pixel values of the
// k matched patches over s^2 (pnlm_flat_sums), alpha =
// clamp((kPnlmFlatHi - r) / (kPnlmFlatHi - kPnlmFlatLo), 0, 1) of the plain
// mean of those pixels is mixed in (the flat-patch rule, softened). The output
// pixel is the tent-weighted mean (pnlm_tent along x times along y) of the
// estimates of every reference patch that covers it, gathered in reference
// order. Every operation is plain IEEE float arithmetic in this order (no
// contraction), so every backend gives the same bits.
#include "nss/params.hpp"

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define NSS_PNLM_HD __host__ __device__
#else
#define NSS_PNLM_HD
#endif

namespace nss {

inline constexpr float kPnlmDefaultSigma = 3.f;
inline constexpr int kPnlmDefaultD = 1;
inline constexpr int kPnlmDefaultA = 2;
inline constexpr int kPnlmDefaultBlock = 6;
inline constexpr int kPnlmDefaultGroup = 16;
// Limits: patches of 4, 6 or 8; groups of 2 to 32; frames of up to 63
// (d <= 31) and planes of up to 8192 x 8192 (the key's fields); windows of
// a <= 64 as NLM's.
inline constexpr int kPnlmMaxD = 31;
inline constexpr int kPnlmMaxA = 64;
inline constexpr int kPnlmMaxGroup = 32;
inline constexpr int kPnlmMaxSide = 8192;
// The weights' scale h = kPnlmLambda s, the distance offset kPnlmBias 2 s^2,
// and the flat-patch ramp over the variance ratio r.
inline constexpr float kPnlmLambda = 2.f;
inline constexpr float kPnlmBias = 1.f;
inline constexpr float kPnlmFlatLo = 0.9f;
inline constexpr float kPnlmFlatHi = 1.6f;

struct PnlmParams {
    float sigma[3]{kPnlmDefaultSigma, kPnlmDefaultSigma, kPnlmDefaultSigma};  // 8-bit units; 0 copies the plane
    int d = kPnlmDefaultD;
    int a = kPnlmDefaultA;
    int block = kPnlmDefaultBlock;
    int group = kPnlmDefaultGroup;
};

// The per-plane constants of the weights and the flat rule, from s.
struct PnlmScale {
    float inv_area;  // 1 / p^2
    float offset;    // kPnlmBias 2 s^2
    float inv_h2;    // 1 / (kPnlmLambda s)^2
    float inv_s2;    // 1 / s^2
};
inline PnlmScale pnlm_scale(float sigma8, int block) {
    const float s = sigma8 / 255.f;
    const float s2 = s * s;
    const float h = kPnlmLambda * s;
    return PnlmScale{1.f / static_cast<float>(block * block), kPnlmBias * (2.f * s2), 1.f / (h * h), 1.f / s2};
}

// The reference grid along one axis of n pixels: positions 0, q, 2q, ... and
// n - p last (count entries; none when n < p).
NSS_PNLM_HD inline int pnlm_grid_count(int n, int block) {
    const int q = block / 2;
    return n < block ? 0 : (n - block + q - 1) / q + 1;
}
NSS_PNLM_HD inline int pnlm_grid_pos(int i, int n, int block) {
    const int v = i * (block / 2), last = n - block;
    return v < last ? v : last;
}

// Search keys: distance bits (capped at +inf), frame, row, column.
NSS_PNLM_HD inline std::uint32_t pnlm_pos(int t, int y, int x) {
    return static_cast<std::uint32_t>(t) << 26 | static_cast<std::uint32_t>(y) << 13 | static_cast<std::uint32_t>(x);
}
NSS_PNLM_HD inline int pnlm_pos_t(std::uint32_t pos) { return static_cast<int>(pos >> 26); }
NSS_PNLM_HD inline int pnlm_pos_y(std::uint32_t pos) { return static_cast<int>((pos >> 13) & 8191u); }
NSS_PNLM_HD inline int pnlm_pos_x(std::uint32_t pos) { return static_cast<int>(pos & 8191u); }

// The q column sums of a quarter, added: (c0 + c1) + (c2 + c3), (c0 + c1) + c2, c0 + c1.
template <int Q>
NSS_PNLM_HD inline float pnlm_columns(const float (&c)[Q]) {
    if constexpr (Q == 4) return (c[0] + c[1]) + (c[2] + c[3]);
    else if constexpr (Q == 3) return (c[0] + c[1]) + c[2];
    else return c[0] + c[1];
}

// exp(x) for x <= 0: 0 below -87 (the result stays a normal float), else
// 2^k 2^f with k = rint(x log2 e), |f| <= 1/2, 2^f by a degree-6 polynomial
// (Cephes exp2f), scaled exactly. Relative error about 2e-7.
NSS_PNLM_HD inline float pnlm_exp(float x) {
    if (!(x > -87.f)) return 0.f;
    const float t = x * 1.44269504088896341f;
    const float k = rintf(t);
    const float f = t - k;
    float p = 1.535336188319500e-4f;
    p = fmaf(p, f, 1.339887440266574e-3f);
    p = fmaf(p, f, 9.618437357674640e-3f);
    p = fmaf(p, f, 5.550332471162809e-2f);
    p = fmaf(p, f, 2.402264791363012e-1f);
    p = fmaf(p, f, 6.931472028550421e-1f);
    p = fmaf(p, f, 1.f);
    return ldexpf(p, static_cast<int>(k));
}

// The weight of a match of distance D (the reference's is 1).
NSS_PNLM_HD inline float pnlm_weight(float dist, const PnlmScale& c) {
    const float u = dist * c.inv_area - c.offset;
    return pnlm_exp(-((u > 0.f ? u : 0.f) * c.inv_h2));
}

// The flat-patch rule: from the sums of the values and of their squares over
// the count = k p^2 pixel values, the mean and its share alpha; an estimate is
// then pnlm_mix(raw, flat) = alpha mean + (1 - alpha) raw, raw = sum w P / sum w.
struct PnlmFlat {
    float alpha_mean;    // alpha mean
    float one_minus;     // 1 - alpha
};
NSS_PNLM_HD inline PnlmFlat pnlm_flat(float s1, float s2, float count, const PnlmScale& c) {
    const float mean = s1 / count;
    const float ex2 = s2 / count;
    const float m2 = mean * mean;
    const float v = ex2 - m2;
    const float r = (v > 0.f ? v : 0.f) * c.inv_s2;
    float alpha = (kPnlmFlatHi - r) / (kPnlmFlatHi - kPnlmFlatLo);
    alpha = alpha < 0.f ? 0.f : (alpha > 1.f ? 1.f : alpha);
    return PnlmFlat{alpha * mean, 1.f - alpha};
}
NSS_PNLM_HD inline float pnlm_mix(float raw, const PnlmFlat& f) {
    const float b = f.one_minus * raw;
    return f.alpha_mean + b;
}

// The flat sums run over pnlm_lanes(p^2) lanes: pixel j of the patch in lane
// j % lanes; each lane adds its pixels match by match (match outer, its
// pixels in increasing j inner; the squares with fmaf), then the lanes are
// added by a butterfly (lane l += lane l ^ o for o = lanes / 2 .. 1).
NSS_PNLM_HD inline constexpr int pnlm_lanes(int area) { return area <= 16 ? 16 : 32; }

// Tent weight of position i in a patch of p: 1 - |2 i + 1 - p| / p.
NSS_PNLM_HD inline float pnlm_tent(int i, int p) {
    const float e = static_cast<float>(2 * i + 1 - p);
    return 1.f - (e < 0.f ? -e : e) / static_cast<float>(p);
}

}  // namespace nss
