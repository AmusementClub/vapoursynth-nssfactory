// SPDX-License-Identifier: GPL-2.0-only
// Device side of FixedTarget (aggregate.hpp): one weighted sample into the
// fixed-point accumulators. Integer addition is associative and commutative,
// so the atomics give the same sums in any order (D17) for any input.
#pragma once

#include "cuda/common/aggregate.hpp"

#include <cuda_runtime.h>

namespace nss_cuda {

// 2^32: a float converts without rounding unless it carries bits below
// 2^-32, and sums up to 2^31 in magnitude fit.
constexpr float kFixedScale = 4294967296.f;

__device__ __forceinline__ void fixed_add(unsigned long long* cell, float value) {
    atomicAdd(cell, static_cast<unsigned long long>(__float2ll_rn(value * kFixedScale)));
}

}  // namespace nss_cuda
