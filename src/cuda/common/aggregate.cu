// SPDX-License-Identifier: GPL-2.0-only
#include "cuda/common/aggregate.hpp"
#include "cuda/runtime/error.hpp"

#include <cub/device/device_radix_sort.cuh>

#include <stdexcept>

namespace nss_cuda {
namespace {

constexpr int kTile = OrderedAggregator::kTile;
constexpr int kTilesPerPatch = 4;  // block <= kTile + 1 spans at most 2x2 tiles
constexpr int kTileThreads = 256;

// Padding entries use key == bins (one past the last bin) so they sort last
// while the radix sort only needs bits_for(bins) bits.
__global__ void expand_kernel(const AggregatePatch* patches, int npatch, int block, int tiles_x, int tiles_y,
                              unsigned invalid, unsigned* keys, int* ids) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= npatch) return;
    const AggregatePatch patch = patches[p];
    const int tx0 = patch.x / kTile, tx1 = min((patch.x + block - 1) / kTile, tiles_x - 1);
    const int ty0 = patch.y / kTile, ty1 = min((patch.y + block - 1) / kTile, tiles_y - 1);
    const unsigned base = static_cast<unsigned>(patch.slice) * static_cast<unsigned>(tiles_x * tiles_y);
    int j = 0;
    for (int ty = ty0; ty <= ty1; ++ty) {
        for (int tx = tx0; tx <= tx1; ++tx, ++j) {
            keys[p * kTilesPerPatch + j] = base + static_cast<unsigned>(ty * tiles_x + tx);
            ids[p * kTilesPerPatch + j] = p;
        }
    }
    for (; j < kTilesPerPatch; ++j) {
        keys[p * kTilesPerPatch + j] = invalid;
        ids[p * kTilesPerPatch + j] = -1;
    }
}

__global__ void bin_bounds_kernel(const unsigned* keys, int n, unsigned invalid, int* begin, int* end) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned key = keys[i];
    if (key == invalid) return;
    if (i == 0 || keys[i - 1] != key) begin[key] = i;
    if (i == n - 1 || keys[i + 1] != key) end[key] = i + 1;
}

// One thread block per (tile, slice); threads own fixed tile pixels and walk
// the bin in patch-id order, so every pixel's sum has a fixed order.
__global__ void __launch_bounds__(kTileThreads)
tile_reduce_kernel(const float* values, const AggregatePatch* patches, const int* ids, const int* begin,
                   const int* end, int block, int tiles_x, int tiles_y, AggregateTarget target) {
    constexpr int kPerThread = kTile * kTile / kTileThreads;
    const int tx = blockIdx.x, ty = blockIdx.y, slice = blockIdx.z;
    const int bin = slice * tiles_x * tiles_y + ty * tiles_x + tx;
    const int col = threadIdx.x % kTile;
    const int row0 = threadIdx.x / kTile;
    const int px = tx * kTile + col;
    float num[kPerThread], den[kPerThread];
    for (int k = 0; k < kPerThread; ++k) num[k] = den[k] = 0.f;
    const int area = block * block;
    for (int e = begin[bin]; e < end[bin]; ++e) {
        const int p = ids[e];
        const AggregatePatch patch = patches[p];
        const int dx = px - patch.x;
        if (dx < 0 || dx >= block) continue;
        const float* v = values + static_cast<long long>(p) * area + dx;
        for (int k = 0; k < kPerThread; ++k) {
            const int dy = ty * kTile + row0 + k * (kTileThreads / kTile) - patch.y;
            if (dy >= 0 && dy < block) {
                num[k] = fmaf(patch.weight, v[dy * block], num[k]);
                den[k] += patch.weight;
            }
        }
    }
    if (px >= target.width) return;
    float* tnum = target.num + slice * target.slice_step;
    float* tden = target.den + slice * target.slice_step;
    for (int k = 0; k < kPerThread; ++k) {
        const int py = ty * kTile + row0 + k * (kTileThreads / kTile);
        if (py < target.height) {
            tnum[static_cast<long long>(py) * target.pitch + px] = num[k];
            tden[static_cast<long long>(py) * target.pitch + px] = den[k];
        }
    }
}

__global__ void atomic_kernel(const float* values, const AggregatePatch* patches, long long total, int block,
                              AggregateTarget target) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int area = block * block;
    const int p = static_cast<int>(i / area);
    const int r = static_cast<int>(i % area) / block, c = static_cast<int>(i % area) % block;
    const AggregatePatch patch = patches[p];
    const long long offset = patch.slice * static_cast<long long>(target.slice_step) +
                             static_cast<long long>(patch.y + r) * target.pitch + patch.x + c;
    atomicAdd(target.num + offset, patch.weight * values[i]);
    atomicAdd(target.den + offset, patch.weight);
}

int bits_for(unsigned value) {
    int bits = 1;
    while (bits < 32 && (value >> bits) != 0) ++bits;
    return bits;
}

std::size_t sort_temp_size(std::size_t entries) {
    std::size_t bytes = 0;
    NSS_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, bytes, static_cast<const unsigned*>(nullptr),
                                                   static_cast<unsigned*>(nullptr), static_cast<const int*>(nullptr),
                                                   static_cast<int*>(nullptr), static_cast<int>(entries)));
    return bytes;
}

}  // namespace

std::size_t OrderedAggregator::workspace_bytes(int width, int height, int slices, int block, std::size_t max_patches) {
    const std::size_t entries = max_patches * kTilesPerPatch;
    const std::size_t bins = static_cast<std::size_t>((width + kTile - 1) / kTile) * ((height + kTile - 1) / kTile) * slices;
    (void)block;
    return entries * (2 * sizeof(unsigned) + 2 * sizeof(int)) + 2 * bins * sizeof(int) + sort_temp_size(entries);
}

OrderedAggregator::OrderedAggregator(int width, int height, int slices, int block, std::size_t max_patches,
                                     const std::shared_ptr<nss::ResourceBudget>& budget)
    : width_(width), height_(height), slices_(slices), block_(block),
      tiles_x_((width + kTile - 1) / kTile), tiles_y_((height + kTile - 1) / kTile), max_patches_(max_patches),
      max_entries_(max_patches * kTilesPerPatch) {
    if (block < 1 || block > kTile + 1) throw std::invalid_argument("nss_cuda: aggregation block exceeds tile size");
    const std::size_t bins = static_cast<std::size_t>(tiles_x_) * tiles_y_ * slices_;
    if (bins >= 0x7fffffffu || max_entries_ > static_cast<std::size_t>(INT32_MAX)) {
        throw std::invalid_argument("nss_cuda: aggregation geometry too large");
    }
    key_bits_ = bits_for(static_cast<unsigned>(bins));
    keys_in_ = DeviceBuffer(max_entries_ * sizeof(unsigned), budget);
    keys_out_ = DeviceBuffer(max_entries_ * sizeof(unsigned), budget);
    ids_in_ = DeviceBuffer(max_entries_ * sizeof(int), budget);
    ids_out_ = DeviceBuffer(max_entries_ * sizeof(int), budget);
    bin_begin_ = DeviceBuffer(bins * sizeof(int), budget);
    bin_end_ = DeviceBuffer(bins * sizeof(int), budget);
    sort_temp_bytes_ = sort_temp_size(max_entries_);
    sort_temp_ = DeviceBuffer(sort_temp_bytes_, budget);
}

void OrderedAggregator::run(const float* values, const AggregatePatch* patches, int npatch,
                            const AggregateTarget& target, cudaStream_t stream) {
    if (static_cast<std::size_t>(npatch) > max_patches_) throw std::logic_error("nss_cuda: aggregation over capacity");
    const std::size_t bins = static_cast<std::size_t>(tiles_x_) * tiles_y_ * slices_;
    NSS_CUDA_CHECK(cudaMemsetAsync(bin_begin_.get(), 0, bins * sizeof(int), stream));
    NSS_CUDA_CHECK(cudaMemsetAsync(bin_end_.get(), 0, bins * sizeof(int), stream));
    const int entries = npatch * kTilesPerPatch;
    if (npatch > 0) {
        const unsigned invalid = static_cast<unsigned>(bins);
        expand_kernel<<<(npatch + 255) / 256, 256, 0, stream>>>(patches, npatch, block_, tiles_x_, tiles_y_, invalid,
                                                                 keys_in_.as<unsigned>(), ids_in_.as<int>());
        NSS_CUDA_CHECK_LAUNCH();
        // Stable LSD radix sort: equal bins keep their input (patch-id) order.
        std::size_t temp = sort_temp_bytes_;
        NSS_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(sort_temp_.get(), temp, keys_in_.as<unsigned>(),
                                                       keys_out_.as<unsigned>(), ids_in_.as<int>(), ids_out_.as<int>(),
                                                       entries, 0, key_bits_, stream));
        bin_bounds_kernel<<<(entries + 255) / 256, 256, 0, stream>>>(keys_out_.as<unsigned>(), entries, invalid,
                                                                     bin_begin_.as<int>(), bin_end_.as<int>());
        NSS_CUDA_CHECK_LAUNCH();
    }
    const dim3 grid(tiles_x_, tiles_y_, slices_);
    tile_reduce_kernel<<<grid, kTileThreads, 0, stream>>>(values, patches, ids_out_.as<int>(), bin_begin_.as<int>(),
                                                          bin_end_.as<int>(), block_, tiles_x_, tiles_y_, target);
    NSS_CUDA_CHECK_LAUNCH();
}

void aggregate_atomic(const float* values, const AggregatePatch* patches, int npatch, int block,
                      const AggregateTarget& target, cudaStream_t stream) {
    const long long total = static_cast<long long>(npatch) * block * block;
    if (total == 0) return;
    atomic_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(values, patches, total, block, target);
    NSS_CUDA_CHECK_LAUNCH();
}

}  // namespace nss_cuda
