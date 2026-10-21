// SPDX-License-Identifier: GPL-2.0-only
#pragma once
// Resolved LSSC arguments and patch-grid geometry shared by every backend.
#include "nss/params.hpp"
#include "nss/checked.hpp"

#include <cstdint>

namespace nss {

struct LsscParams {
    float sigma[3]{kLsscDefaultSigma, kLsscDefaultSigma, kLsscDefaultSigma};
    int block_size = kLsscDefaultBlock;
    int block_step = kLsscDefaultStep;
    int radius = 0;
};

// Model version 4, every backend: each grid patch is coded on its own by
// error-constrained orthogonal matching pursuit (Batch-OMP over the Gram matrix
// G = D^T D) against the plane's dictionary, and the reconstructions are
// averaged with the noisy plane added at a small weight. All in FP32, every
// fused operation explicit. Sums written "lane sum" are the device's warp
// sum: 32 lanes, lane l adding terms l, l + 32, ... from 0 in order, then a
// butterfly over lane distances 16, 8, 4, 2, 1.
//   mean = lane sum of the pixels / m, y = patch - mean, ||y||^2 = lane sum
//   of y^2, alpha0 = D^T y (one fma chain over the pixels per atom), alpha =
//   alpha0;
//   until ||r||^2 <= lssc_omp_budget or kLsscOmpMaxAtoms atoms:
//     pick the unused atom of the largest |alpha_a| / ||d_a|| (lowest index on
//     ties); the new factor row L_k = L^-1 g (g_t = G[best][S_t], reciprocal
//     diagonals), the pivot G_bb - lane sum of L_k^2 (stop when it is not
//     above kLsscOmpPivot G_bb), z_k = (alpha0_best - lane sum of L_k z) /
//     L_kk, ||r||^2 -= z_k^2; then (unless this was the last atom) u_k =
//     (G[best] - sum_q L_kq u_q) / L_kk (one fma chain over q per atom; u_t =
//     G_S L^-T e_t) and alpha -= z_k u_k;
//   x = L^-T z (row t subtracts the later rows in descending order),
//   patch = mean + D_S x (fma chains in support order),
//   output = (sum of patches + w noisy) / (count + w),
//   w = kLsscImageWeight (block / step)^2 / sigma8 (the K-SVD image term).
// The coefficients are least squares, not shrunk: model version 3's group
// soft threshold dimmed high-contrast detail in proportion to sigma (white
// text by about 4 levels at the default sigma). The defaults need overlap
// (block_step 4); docs/lssc-fix-eval-20261021.md has the evaluation.
inline constexpr float kLsscOmpResidual = 1.15f;
inline constexpr int kLsscOmpMaxAtoms = 48;
inline constexpr float kLsscOmpPivot = 1e-4f;
inline constexpr float kLsscImageWeight = 2.5f;
// The default reference stride: half the block, so that every pixel is
// covered by about four patches (the image weight assumes coverage).
inline int lssc_default_step(int block) { return block > 1 ? block / 2 : 1; }
// The weight of the noisy plane in the average (sigma normalized to [0, 1]).
inline float lssc_image_weight(int block, int step, float sigma) {
    const float coverage = static_cast<float>(block * block) / static_cast<float>(step * step);
    return kLsscImageWeight * coverage / (sigma * 255.f);
}
// The residual energy at which a patch's coding stops: m (C sigma)^2.
inline float lssc_omp_budget(int m, float sigma) {
    const float c = kLsscOmpResidual * sigma;
    return static_cast<float>(m) * (c * c);
}
// Atoms a patch of m pixels may take from a dictionary of `atoms`.
inline int lssc_omp_cap(int m, int atoms) {
    const int cap = m < atoms ? m : atoms;
    return cap < 0 ? 0 : cap < kLsscOmpMaxAtoms ? cap : kLsscOmpMaxAtoms;
}

// Model version 3's cluster coding, kept for the library users of
// lssc_reconstruct (tests and the paper-track explorer); nss.LSSC no longer
// runs it.
// Group sparse coding of a cluster (model version 3):
// accelerated proximal gradient (FISTA) over kLsscCodingRounds rounds on the
// preconditioned problem
//   0.5 (Y - D A)^T M (Y - D A) + sum_a lambda_a ||A_a,:||_2,
//   M = c (D D^T + delta I)^-1, delta = kLsscPrecondAlpha * s, c = s + delta,
// where s is the mean eigenvalue of D D^T (trace / m): M weighs a direction of
// eigenvalue e by (s + delta) / (e + delta), 1 at the mean. The gradient uses
// P = D^T M in place of D^T, the step is 0.9 / L' with
// L' = c L / (L + delta) from the power estimate L of ||D||^2, and
// lambda_a = (sigma / 255) sqrt(cluster size) ||M d_a||. Round r extrapolates
// with kLsscMomentum[r] (the FISTA t-sequence from t = 1, as floats).
inline constexpr int kLsscCodingRounds = 12;
inline constexpr float kLsscPrecondAlpha = 3.f;
inline constexpr float kLsscMomentum[kLsscCodingRounds - 1] = {
    0.f,          0.28175351f,  0.434042782f, 0.531063795f, 0.598778605f, 0.648923337f,
    0.687645853f, 0.718499839f, 0.743691027f, 0.76466471f,  0.782409251f};

inline int lssc_axis_count(int len, int block, int step) {
    if (len < block || block < 1 || step < 1) {
        return 0;
    }
    return checked_int((static_cast<std::uint64_t>(len - block) + step - 1) / step + 1);
}

inline int lssc_grid_count(int width, int height, int block, int step) {
    return checked_int(static_cast<std::uint64_t>(lssc_axis_count(width, block, step)) * lssc_axis_count(height, block, step));
}

}  // namespace nss
