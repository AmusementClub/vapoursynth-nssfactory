// SPDX-License-Identifier: GPL-2.0-only
// Host side of the Gram group kernels (gram_group.cuh): what a filter asks
// the driver for.
#pragma once

#include <cstddef>

namespace nss_cuda {

// Scratch per group of up to 8 patches: the Gram matrix's lower triangle, M
// and the aggregation weight, handed from kernel to kernel.
inline constexpr int kGramSplitGram = 36, kGramSplitFloats = kGramSplitGram + 64 + 1;

// GroupFilterConfig::scratch_floats of a filter on the Gram kernels.
inline std::size_t gram_scratch_floats(int group) {
    return group <= 8 ? static_cast<std::size_t>(kGramSplitFloats) : 0;
}

}  // namespace nss_cuda
