// SPDX-License-Identifier: GPL-2.0-only
// MCWNNM group filter on the device (the CPU mcwnnm_filter_group model): the
// three channels of every matched patch are stacked into one m x n matrix
// (m = 3 block^2), rows are optionally centered, and the channel-weighted
// WNNM objective is solved by ADMM (X/Z/A iterates, rho growing by mu).
//
// Each Z step is the WNNM shrinkage of X + A / rho, taken through the FP32
// Gram matrix (common/gram_shrink.cuh). Everything after the eigen step is
// local to a matrix row, so one pass over the rows applies Z, the dual and
// the next X and accumulates the next Gram matrix; Z is never stored.
#pragma once

#include "cuda/common/aggregate.hpp"
#include "cuda/common/match.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace nss_cuda {

struct McwnnmGroupArgs {
    const float* const* src;    // device array of per-frame pointers (DeviceMatch::t), 3 stacked channels
    int pitch;
    long long channel_step;     // floats between channels of a frame
    const DeviceMatch* matches; // batch * group (group-strided)
    const int* counts;
    int batch;
    int block;
    int group;
    float sigma[3];             // user sigma / 255 per channel
    int residual;
    int adaptive;
    int admm_iter;
    float rho;
    float mu;
    float* values;              // 3 * batch * group * block^2, channel-major
    float* scratch;             // batch * mcwnnm_scratch_floats(block, group)
    AggregatePatch* patches;    // batch * group
};

// X and A.
inline std::size_t mcwnnm_scratch_floats(int block, int group) {
    return 2 * static_cast<std::size_t>(3 * block * block) * group;
}

void mcwnnm_filter_groups(const McwnnmGroupArgs& args, cudaStream_t stream);

}  // namespace nss_cuda
