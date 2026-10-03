// SPDX-License-Identifier: GPL-2.0-only
// NCSR group filter on the device (the CPU ncsr_filter_group model): PCA of
// the row-centered m x n group matrix, each code row pulled toward its
// distance-weighted mean and soft-thresholded around it with the adaptive
// threshold 2 sqrt(2) sigma^2 / (weighted code deviation), then reconstruction.
//
// The PCA is taken through the FP32 Gram matrix: A = U S V^T, so the code row
// of component i is s_i V[:, i]^T and the result is A (V S^-1 B') with B' the
// shrunk codes. Components with a zero singular value carry zero codes and
// drop out, as on the CPU.
#pragma once

#include "cuda/common/aggregate.hpp"
#include "cuda/common/match.hpp"

#include <cuda_runtime.h>

namespace nss_cuda {

struct NcsrGroupArgs {
    const float* const* src;    // device array of per-frame plane pointers (DeviceMatch::t)
    int pitch;
    const DeviceMatch* matches; // batch * group (group-strided); dist feeds the weights
    const int* counts;
    int batch;
    int block;
    int group;
    float sigma;                // user sigma / 255
    float* values;              // batch * group * block^2
    AggregatePatch* patches;    // batch * group
};

void ncsr_filter_groups(const NcsrGroupArgs& args, cudaStream_t stream);

}  // namespace nss_cuda
