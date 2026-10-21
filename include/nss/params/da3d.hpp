// SPDX-License-Identifier: GPL-2.0-only
#pragma once
// Resolved DA3D arguments, constants and tile geometry shared by every backend.
//
// DA3D (Pierazzo and Facciolo, "Data Adaptive Dual Domain Denoising: a Method
// to Boost State of the Art Denoising Algorithms", IPOL 7 (2017) 93-114),
// model version 1. A last step on top of another denoiser: the noisy plane y
// is filtered with the help of a guide g (the other denoiser's output).
// The image is cut into a fixed tile grid (da3d_tiles); every tile is
// processed on its own, padded by kDa3dRadius before and kDa3dPatch -
// kDa3dRadius - 1 after with the plane mirrored at its borders, and the tiles'
// sums are added in tile order. Inside a tile, until every core pixel has an
// aggregation weight of at least kDa3dThreshold:
//   p = the core pixel of least weight (quad-tree min pyramid; the first of a
//   2x2 square in the order (r, c), (r+1, c), (r, c+1), (r+1, c+1) wins
//   ties), the 64x64 patches y and g with p at (31, 31);
//   k_reg = fastexp(-(|g - g(p)|^2 / (10 gamma_r sigma^2) + d^2 / (4 sigma_s^2)))
//   with d the distance to p; the plane through g(p) of least k_reg-weighted
//   squared error to y (none when the 2x2 system's determinant is below
//   1e-4) is subtracted from y and g;
//   k = fastexp(-(|g - g(p)|^2 / (gamma_r sigma^2) + d^2 / (2 sigma_s^2)));
//   if sum k < kDa3dGuideOnlyMass: add (g + plane) k^2 and k^2;
//   else: avg = sum k y / sum k, y_m = k y + (1 - k) avg (and g_m alike),
//   Y, G = DFT(y_m), DFT(g_m) (unnormalized), every coefficient but DC
//   times fastexp(-gamma_f sigma^2 sum k^2 / |G|^2), y' = IDFT / 4096,
//   add ((y' - (1 - k) avg) + plane k) k and k^2;
//   the weight map gains k^2.
// The output is the sum over the tiles divided by the weight sum. |.|^2 sums
// the channels: RGB is filtered as one 3-channel image after the orthonormal
// transform ((r+g+b)/sqrt3, (r-b)/sqrt2, (r-2g+b)/sqrt6) and transformed back;
// any other plane is filtered on its own. fastexp is the reference's
// Schraudolph exponential (da3d_fastexp). Every patch sum is the 32-lane sum of
// LsscLaneSum (lane l takes the terms l, l + 32, ... in order, then a halving
// tree), so the result does not depend on the vector width. The tile grid is
// part of the model: another grid gives the same quality, not the same pixels.
#include "nss/params.hpp"

#include <cstdint>
#include <cstring>

namespace nss {

inline constexpr int kDa3dRadius = 31;    // r: p sits at (r, r) of the patch
inline constexpr int kDa3dPatch = 64;     // s = NextPowerOf2(2 r + 1)
inline constexpr int kDa3dTile = 128;     // the tile grid's nominal edge
inline constexpr float kDa3dSigmaS = 14.f;
inline constexpr float kDa3dGammaR = 0.7f;
inline constexpr float kDa3dGammaF = 0.8f;
inline constexpr float kDa3dThreshold = 2.f;
inline constexpr float kDa3dGuideOnlyMass = 10.f;
inline constexpr float kDa3dRegressionRangeScale = 10.f;   // gamma_rr = 10 gamma_r
inline constexpr float kDa3dRegressionSpatialScale = 2.f;  // sigma_sr^2 = 2 sigma_s^2
inline constexpr float kDa3dRegressionEpsilon = 1e-4f;
inline constexpr float kDa3dDefaultSigma = 3.f;

struct Da3dParams {
    float sigma[3]{kDa3dDefaultSigma, kDa3dDefaultSigma, kDa3dDefaultSigma};  // 8-bit units, true noise
};

// The reference's exponential: exp(x) read off the float bit pattern
// 12102203 x + 127 * 2^23, truncated toward zero, and 0 where that is not
// positive (x below about -87.3, including -inf).
inline float da3d_fastexp(float x) {
    const float t = 12102203.f * x;
    if (!(t > -1065353216.f)) return 0.f;
    const std::int32_t bits = static_cast<std::int32_t>(t) + 1065353216;
    if (bits <= 0) return 0.f;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// Tiles along an edge of n pixels: n / kDa3dTile rounded to nearest, at least
// one. Tile i covers [n i / count, n (i + 1) / count).
inline int da3d_tile_count(int n) {
    const int count = (n + kDa3dTile / 2) / kDa3dTile;
    return count > 0 ? count : 1;
}
inline int da3d_tile_start(int n, int count, int i) {
    return static_cast<int>(static_cast<std::int64_t>(n) * i / count);
}
// The largest tile edge of the grid along n.
inline int da3d_tile_max(int n) {
    const int count = da3d_tile_count(n);
    int best = 0;
    for (int i = 0; i < count; ++i) {
        const int e = da3d_tile_start(n, count, i + 1) - da3d_tile_start(n, count, i);
        best = e > best ? e : best;
    }
    return best;
}

}  // namespace nss
