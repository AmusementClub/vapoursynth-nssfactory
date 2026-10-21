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
// averaged with the noisy plane added at a small weight:
//   y = patch - mean (FP32; mean the FP64 sum / m), ||y||^2 in FP64,
//   alpha0 = D^T y, alpha = alpha0 (FP32, one fma chain over the pixels each),
//   until ||r||^2 <= m (kLsscOmpResidual sigma)^2 or kLsscOmpMaxAtoms atoms:
//     pick the unused atom of the largest |alpha_a| / ||d_a|| (lowest index on
//     ties), extend the FP64 Cholesky factor of G over the support (stop when
//     the new pivot is not above kLsscOmpPivot G_aa), its forward solution
//     z (L z = alpha0_S, one new entry per atom), ||r||^2 = ||y||^2 - ||z||^2,
//     x = L^-T z (row t subtracts the later rows in descending order), all
//     FP64; alpha = alpha0 - G_S float(x),
//   patch = mean + D_S float(x) (FP32 fma chains in support order),
//   output = (sum of patches + w noisy) / (count + w),
//   w = kLsscImageWeight (block / step)^2 / sigma8 (the K-SVD image term).
// The coefficients are least squares, not shrunk: model version 3's group
// soft threshold dimmed high-contrast detail in proportion to sigma (white
// text by about 4 levels at the default sigma). The defaults need overlap
// (block_step 4); docs/lssc-fix-eval-20261021.md has the evaluation.
inline constexpr float kLsscOmpResidual = 1.15f;
inline constexpr int kLsscOmpMaxAtoms = 48;
inline constexpr float kLsscOmpPivot = 1e-6f;
inline constexpr float kLsscImageWeight = 2.5f;

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
