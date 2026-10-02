// SPDX-License-Identifier: GPL-2.0-only
// Weighted patch aggregation into num/den planes (num += w * v, den += w per
// covered pixel, as nss::unpack_patch), optionally into 2R+1 temporal slices.
//
// OrderedAggregator is deterministic (D17): patches are binned per output
// tile with a stable radix sort keyed by tile, so each tile sums its patches
// in patch-id order. Callers number patches in the CPU commit order (raster
// reference index, then group position), which then is also the CPU's
// per-pixel accumulation order. aggregate_atomic is the float-atomic
// alternative kept for the D17 measurement; its sums depend on scheduling.
#pragma once

#include "cuda/runtime/memory.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>

namespace nss_cuda {

struct AggregatePatch {
    int x;
    int y;
    int slice;  // 0 for spatial output, [0, 2R] for temporal fat output
    float weight;
};

struct AggregateTarget {
    float* num;
    float* den;
    int width;
    int height;
    int pitch;               // floats
    int slices;
    std::size_t slice_step;  // floats between slices of num (and of den)
};

class OrderedAggregator {
public:
    static constexpr int kTile = 32;
    OrderedAggregator(int width, int height, int slices, int block, std::size_t max_patches,
                      const std::shared_ptr<nss::ResourceBudget>& budget);
    // Overwrites every pixel of every slice of target (no prior zeroing needed).
    // values: npatch * block * block floats, patch-major, row-major inside a patch.
    void run(const float* values, const AggregatePatch* patches, int npatch, const AggregateTarget& target,
             cudaStream_t stream);
    static std::size_t workspace_bytes(int width, int height, int slices, int block, std::size_t max_patches);

private:
    int width_, height_, slices_, block_;
    int tiles_x_, tiles_y_;
    std::size_t max_patches_, max_entries_;
    DeviceBuffer keys_in_, keys_out_, ids_in_, ids_out_, bin_begin_, bin_end_, sort_temp_;
    std::size_t sort_temp_bytes_ = 0;
    int key_bits_ = 0;
};

// Adds into target (which the caller zeroes); nondeterministic summation order.
void aggregate_atomic(const float* values, const AggregatePatch* patches, int npatch, int block,
                      const AggregateTarget& target, cudaStream_t stream);

}  // namespace nss_cuda
