// SPDX-License-Identifier: GPL-2.0-only
// Device side of FixedTarget (aggregate.hpp): one weighted sample into the
// fixed-point accumulators. Integer addition is associative and commutative,
// so the atomics give the same sums in any order for any input.
#pragma once

#include "cuda/common/aggregate.hpp"
#include "gpuhost/fixed.hpp"

#include <cuda_runtime.h>

namespace nss_cuda {

using nss::gpuhost::kFixedScale;  // 2^32 (gpuhost/fixed.hpp)

// Ring cell of the slice of a patch of window slot t, or -1 when the target
// has none for it.
__device__ __forceinline__ int fixed_slice(const FixedTarget& target, int t) {
    const int slice = t + target.slice_base;
    if (static_cast<unsigned>(slice) >= static_cast<unsigned>(target.slices)) return -1;
    const int cell = slice + target.slice_first;
    return cell < target.slice_ring ? cell : cell - target.slice_ring;
}

// First accumulator cell of a patch at (x, y) of window slot t, or -1 when
// the target keeps no slice for that frame.
__device__ __forceinline__ long long fixed_patch_cell(const FixedTarget& target, int t, int y, int x) {
    const int slice = fixed_slice(target, t);
    if (slice < 0) return -1;
    return slice * static_cast<long long>(target.slice_step) + static_cast<long long>(y) * target.pitch + x;
}

__device__ __forceinline__ void fixed_add(unsigned long long* cells, long long at, float value) {
    atomicAdd(cells + at, static_cast<unsigned long long>(__float2ll_rn(value * kFixedScale)));
}

}  // namespace nss_cuda
