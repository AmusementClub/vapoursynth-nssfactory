// SPDX-License-Identifier: GPL-2.0-only
// Host side of the Gram group kernels (gram_group.cuh): what a filter asks
// the driver for.
#pragma once

#include <cstddef>

namespace nss_cuda {

// Scratch per group of up to 8 patches: the Gram matrix's lower triangle, M
// and the aggregation weight, handed from kernel to kernel.
inline constexpr int kGramSplitGram = 36, kGramSplitFloats = kGramSplitGram + 64 + 1;

// Scratch per group of 17 to 32 patches for a model whose eigen step is
// divide and conquer (block_eigen_dc_reals(32), checked in gram_group.cuh).
inline constexpr int kGramDcFloats = 3 * 32 * 32 + 20 * 32 + 8;

// GroupFilterConfig::scratch_floats of a filter on the Gram kernels;
// `divide` for a model with kDivideAndConquer.
inline std::size_t gram_scratch_floats(int group, bool divide = false) {
    if (group <= 8) return static_cast<std::size_t>(kGramSplitFloats);
    return divide && group > 16 ? static_cast<std::size_t>(kGramDcFloats) : 0;
}

}  // namespace nss_cuda
