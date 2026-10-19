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

// Group sparse coding of a cluster (model version 3, every backend):
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
