// SPDX-License-Identifier: GPL-2.0-only
// Weighted patch aggregation into num/den planes (num += w * v, den += w per
// covered pixel, as nss::unpack_patch), optionally into 2R+1 temporal slices.
//
// OrderedAggregator is deterministic (D17): patches are binned per output
// tile with a stable radix sort keyed by tile, so each tile sums its patches
// in patch-id order. Callers number patches in the CPU commit order (raster
// reference index, then group position), which then is also the CPU's
// per-pixel accumulation order. FixedTarget is the other deterministic
// route, for kernels that aggregate their own output: exact fixed-point sums
// by integer atomics. aggregate_atomic is the float-atomic alternative kept
// for the D17 measurement; its sums depend on scheduling.
#pragma once

#include "cuda/runtime/memory.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>

namespace nss_cuda {

struct AggregatePatch {
    int x;
    int y;
    int slice;  // 0 for spatial output, [0, 2R] for temporal fat output; < 0 = empty slot (skipped)
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

// Capacity-based: one instance serves every plane/geometry up to the
// constructed maxima (target size and slices, patch count).
class OrderedAggregator {
public:
    static constexpr int kTile = 32;
    static constexpr int kMaxBlock = kTile + 1;
    // fine_tiles: 8x8 tiles for blocks up to 8 (fewer entries tested per
    // pixel, four times the bins); otherwise 16x16 up to 16. bin_bytes() is
    // what the bin arrays of a given geometry take, for the caller's plan.
    OrderedAggregator(int max_width, int max_height, int max_slices, std::size_t max_patches,
                      const std::shared_ptr<nss::ResourceBudget>& budget, bool fine_tiles = false);
    static std::size_t bin_bytes(int max_width, int max_height, int max_slices, bool fine_tiles);
    // values: npatch * block * block floats, patch-major, row-major inside a
    // patch. accumulate=false overwrites every pixel of every slice of target;
    // accumulate=true adds this batch after the target's current sums, so
    // batches run in patch-id order keep the per-pixel summation order.
    // With pixel_den, den += pixel_den[(patch id / den_group) * block^2 + pixel]
    // instead of the patch weight (per-pixel counts shared by the den_group
    // patches of one group, as NLH's pixel matrices).
    void run(const float* values, const AggregatePatch* patches, int npatch, int block, const AggregateTarget& target,
             cudaStream_t stream, bool accumulate = false, const float* pixel_den = nullptr, int den_group = 1);
    // Device bytes per patch of capacity (sort keys/ids, CUB alternate buffers).
    static constexpr std::size_t kBytesPerPatch = 4 * (4 * sizeof(unsigned) + 4 * sizeof(int));

private:
    bool fine_tiles_;
    std::size_t max_bins_, max_patches_, max_entries_;
    DeviceBuffer keys_in_, keys_out_, ids_in_, ids_out_, bin_begin_, bin_end_, sort_temp_;
    std::size_t sort_temp_bytes_ = 0;
};

// Accumulators for kernels that aggregate their own output (see
// fixed_accumulate.cuh): fixed-point sums, 2^32 units per 1.0, wrapped in
// unsigned 64-bit cells. The sums are exact integers, so they do not depend
// on the order of the atomics. num takes weight * value at every pixel of a
// patch. den takes the weight once, at the patch's top-left pixel: every
// pixel of a patch has the same weight, so the den of a pixel is the sum of
// those cells over the block x block positions that cover it, which
// fixed_finish takes. That halves the atomics and the memory they touch (the
// kernels are bound by it), and the integer sums stay exact.
// Slice s of a plane starts slice_step cells
// after slice s - 1. A sum wraps beyond 2^31 in magnitude and a NaN sample
// adds nothing; both are far outside video ranges and stay deterministic.
// A patch of window slot t goes to slice t + slice_base and is dropped when
// that is outside [0, slices): the rolling driver keeps one slice per frame
// of its chunk and drops what a center contributes to other frames.
struct FixedTarget {
    unsigned long long* num = nullptr;
    unsigned long long* den = nullptr;
    int pitch = 0;               // cells per row
    std::size_t slice_step = 0;  // cells between slices
    int slice_base = 0;
    int slices = 1;
};

// out = den > 1e-12 ? num / den : src, straight from the accumulators of one
// slice (aggregate_finish without the float planes) for patches of
// block x block pixels. rows is scratch of width * height cells (pitch as
// the target).
void fixed_finish(const FixedTarget& target, int block, const float* src, int width, int height,
                  unsigned long long* rows, float* out, cudaStream_t stream);

// Zeroes `count` cells of num and den.
void fixed_clear(const FixedTarget& target, std::size_t count, cudaStream_t stream);

// Adds into target (which the caller zeroes); nondeterministic summation order.
void aggregate_atomic(const float* values, const AggregatePatch* patches, int npatch, int block,
                      const AggregateTarget& target, cudaStream_t stream);

// out = den > 1e-12 ? num / den : src (nss::aggregate_finish).
void aggregate_finish(const float* num, const float* den, const float* src, int width, int height, int pitch,
                      float* out, cudaStream_t stream);

// acc_num += num, acc_den += den over `count` floats (rolling accumulation,
// one call per center in ascending center order as the CPU rolling path).
void accumulate_slice(float* acc_num, float* acc_den, const float* num, const float* den, std::size_t count,
                      cudaStream_t stream);

// x += delta * (y - x) over `count` floats (nss::iter_regularize): relaxes the
// current estimate toward the noisy input between outer iterations.
void iter_regularize(float* x, const float* y, std::size_t count, float delta, cudaStream_t stream);

}  // namespace nss_cuda
