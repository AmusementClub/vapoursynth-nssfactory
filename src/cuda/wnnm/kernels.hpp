// SPDX-License-Identifier: GPL-2.0-only
// WNNM group shrinkage on the device (the CPU wnnm_shrink model): the m x n
// group matrix (m = block^2 rows, n = matched patches as columns), optional
// row centering, singular values shrunk by (s + sqrt(s^2 - c)) / 2 with
// c = 8 sqrt(2n) sigma^2 walking the ordered spectrum, reconstruction, and
// the adaptive weight 1 / kept.
//
// The SVD is taken through the Gram matrix: A^T A = V S^2 V^T by cyclic
// Jacobi, X = A (V diag(s'/s) V^T), all in FP32. The D15 study measured the
// FP64 Gram/eigen variant at 0.2-0.8x the CPU with no gate benefit (FP32 is
// 86-143 dB against nss.WNNM on the frozen references).
#pragma once

#include "cuda/common/aggregate.hpp"
#include "cuda/common/match.hpp"

#include <cuda_runtime.h>

namespace nss_cuda {

struct WnnmGroupArgs {
    const float* const* src;    // device array of per-frame plane pointers (DeviceMatch::t)
    int pitch;
    const DeviceMatch* matches; // batch * group (group-strided)
    const int* counts;
    int batch;
    int block;
    int group;
    float sigma;                // user sigma / 255
    int residual;
    int adaptive;
    float* values;              // batch * group * block^2
    AggregatePatch* patches;    // batch * group
};

void wnnm_filter_groups(const WnnmGroupArgs& args, cudaStream_t stream);

}  // namespace nss_cuda
