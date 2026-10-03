// SPDX-License-Identifier: GPL-2.0-only
#include "cuda/common/aggregate.hpp"
#include "cuda/runtime/error.hpp"

#include <cub/device/device_radix_sort.cuh>

#include <stdexcept>

namespace nss_cuda {
namespace {

// Tiles are 16x16 for blocks up to 16 (one pixel per thread, fewer
// redundant coverage tests) and 32x32 above; a block <= tile + 1 spans at
// most 2x2 tiles either way.
constexpr int kSmallTile = 16;
constexpr int kLargeTile = OrderedAggregator::kTile;
constexpr int kTilesPerPatch = 4;
__host__ __device__ constexpr int tile_for(int block) { return block <= kSmallTile ? kSmallTile : kLargeTile; }
constexpr int kTileThreads = 256;

// Padding entries use key == bins (one past the last bin) so they sort last
// while the radix sort only needs bits_for(bins) bits.
__global__ void expand_kernel(const AggregatePatch* patches, int npatch, int block, int tile, int tiles_x, int tiles_y,
                              unsigned invalid, unsigned* keys, int* ids) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= npatch) return;
    const AggregatePatch patch = patches[p];
    if (patch.slice < 0) {
        for (int j = 0; j < kTilesPerPatch; ++j) {
            keys[p * kTilesPerPatch + j] = invalid;
            ids[p * kTilesPerPatch + j] = -1;
        }
        return;
    }
    const int tx0 = patch.x / tile, tx1 = min((patch.x + block - 1) / tile, tiles_x - 1);
    const int ty0 = patch.y / tile, ty1 = min((patch.y + block - 1) / tile, tiles_y - 1);
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
// the bin in patch-id order (patch records staged through shared memory), so
// every pixel's sum has a fixed order.
template <int kTile>
__global__ void __launch_bounds__(kTileThreads)
tile_reduce_kernel(const float* values, const AggregatePatch* patches, const int* ids, const int* begin,
                   const int* end, int block, int tiles_x, int tiles_y, AggregateTarget target, bool accumulate) {
    constexpr int kPerThread = kTile * kTile / kTileThreads;
    constexpr int kRowStep = kTileThreads / kTile;
    __shared__ AggregatePatch staged[kTileThreads];
    __shared__ int staged_id[kTileThreads];
    const int tx = blockIdx.x, ty = blockIdx.y, slice = blockIdx.z;
    const int bin = slice * tiles_x * tiles_y + ty * tiles_x + tx;
    const int col = threadIdx.x % kTile;
    const int row0 = threadIdx.x / kTile;
    const int px = tx * kTile + col;
    float* tnum = target.num + slice * target.slice_step;
    float* tden = target.den + slice * target.slice_step;
    float num[kPerThread], den[kPerThread];
    for (int k = 0; k < kPerThread; ++k) {
        const int py = ty * kTile + row0 + k * kRowStep;
        const bool inside = accumulate && px < target.width && py < target.height;
        num[k] = inside ? tnum[static_cast<long long>(py) * target.pitch + px] : 0.f;
        den[k] = inside ? tden[static_cast<long long>(py) * target.pitch + px] : 0.f;
    }
    const int area = block * block;
    const int first = begin[bin], last = end[bin];
    for (int chunk = first; chunk < last; chunk += kTileThreads) {
        const int n = min(kTileThreads, last - chunk);
        __syncthreads();
        if (threadIdx.x < n) {
            const int p = ids[chunk + threadIdx.x];
            staged_id[threadIdx.x] = p;
            staged[threadIdx.x] = patches[p];
        }
        __syncthreads();
        for (int e = 0; e < n; ++e) {
            const AggregatePatch patch = staged[e];
            const int dx = px - patch.x;
            if (dx < 0 || dx >= block) continue;
            const float* v = values + static_cast<long long>(staged_id[e]) * area + dx;
#pragma unroll
            for (int k = 0; k < kPerThread; ++k) {
                const int dy = ty * kTile + row0 + k * kRowStep - patch.y;
                if (dy >= 0 && dy < block) {
                    num[k] = fmaf(patch.weight, v[dy * block], num[k]);
                    den[k] += patch.weight;
                }
            }
        }
    }
    if (px >= target.width) return;
    for (int k = 0; k < kPerThread; ++k) {
        const int py = ty * kTile + row0 + k * kRowStep;
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
    if (patch.slice < 0) return;
    const long long offset = patch.slice * static_cast<long long>(target.slice_step) +
                             static_cast<long long>(patch.y + r) * target.pitch + patch.x + c;
    atomicAdd(target.num + offset, patch.weight * values[i]);
    atomicAdd(target.den + offset, patch.weight);
}

__global__ void finish_kernel(const float* num, const float* den, const float* src, int width, int height, int pitch,
                              float* out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const long long i = static_cast<long long>(y) * pitch + x;
    const float d = den[i];
    out[i] = d > 1e-12f ? num[i] / d : src[i];
}

__global__ void accumulate_slice_kernel(float* acc_num, float* acc_den, const float* num, const float* den,
                                        std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    acc_num[i] += num[i];
    acc_den[i] += den[i];
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

OrderedAggregator::OrderedAggregator(int max_width, int max_height, int max_slices, std::size_t max_patches,
                                     const std::shared_ptr<nss::ResourceBudget>& budget)
    : max_bins_(static_cast<std::size_t>((max_width + kSmallTile - 1) / kSmallTile) *
                ((max_height + kSmallTile - 1) / kSmallTile) * max_slices),
      max_patches_(max_patches), max_entries_(max_patches * kTilesPerPatch) {
    if (max_bins_ >= 0x7fffffffu || max_entries_ > static_cast<std::size_t>(INT32_MAX)) {
        throw std::invalid_argument("nss_cuda: aggregation geometry too large");
    }
    keys_in_ = DeviceBuffer(max_entries_ * sizeof(unsigned), budget);
    keys_out_ = DeviceBuffer(max_entries_ * sizeof(unsigned), budget);
    ids_in_ = DeviceBuffer(max_entries_ * sizeof(int), budget);
    ids_out_ = DeviceBuffer(max_entries_ * sizeof(int), budget);
    bin_begin_ = DeviceBuffer(max_bins_ * sizeof(int), budget);
    bin_end_ = DeviceBuffer(max_bins_ * sizeof(int), budget);
    sort_temp_bytes_ = sort_temp_size(max_entries_);
    sort_temp_ = DeviceBuffer(sort_temp_bytes_, budget);
}

void OrderedAggregator::run(const float* values, const AggregatePatch* patches, int npatch, int block,
                            const AggregateTarget& target, cudaStream_t stream, bool accumulate) {
    const int tile = tile_for(block);
    const int tiles_x = (target.width + tile - 1) / tile;
    const int tiles_y = (target.height + tile - 1) / tile;
    const std::size_t bins = static_cast<std::size_t>(tiles_x) * tiles_y * target.slices;
    if (static_cast<std::size_t>(npatch) > max_patches_ || bins > max_bins_ || block < 1 || block > kMaxBlock) {
        throw std::logic_error("nss_cuda: aggregation over capacity");
    }
    NSS_CUDA_CHECK(cudaMemsetAsync(bin_begin_.get(), 0, bins * sizeof(int), stream));
    NSS_CUDA_CHECK(cudaMemsetAsync(bin_end_.get(), 0, bins * sizeof(int), stream));
    const int entries = npatch * kTilesPerPatch;
    if (npatch > 0) {
        const unsigned invalid = static_cast<unsigned>(bins);
        expand_kernel<<<(npatch + 255) / 256, 256, 0, stream>>>(patches, npatch, block, tile, tiles_x, tiles_y,
                                                                 invalid, keys_in_.as<unsigned>(), ids_in_.as<int>());
        NSS_CUDA_CHECK_LAUNCH();
        // Stable LSD radix sort: equal bins keep their input (patch-id) order.
        std::size_t temp = sort_temp_bytes_;
        NSS_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(sort_temp_.get(), temp, keys_in_.as<unsigned>(),
                                                       keys_out_.as<unsigned>(), ids_in_.as<int>(), ids_out_.as<int>(),
                                                       entries, 0, bits_for(invalid), stream));
        bin_bounds_kernel<<<(entries + 255) / 256, 256, 0, stream>>>(keys_out_.as<unsigned>(), entries, invalid,
                                                                     bin_begin_.as<int>(), bin_end_.as<int>());
        NSS_CUDA_CHECK_LAUNCH();
    }
    const dim3 grid(tiles_x, tiles_y, target.slices);
    if (tile == kSmallTile) {
        tile_reduce_kernel<kSmallTile><<<grid, kTileThreads, 0, stream>>>(
            values, patches, ids_out_.as<int>(), bin_begin_.as<int>(), bin_end_.as<int>(), block, tiles_x, tiles_y,
            target, accumulate);
    } else {
        tile_reduce_kernel<kLargeTile><<<grid, kTileThreads, 0, stream>>>(
            values, patches, ids_out_.as<int>(), bin_begin_.as<int>(), bin_end_.as<int>(), block, tiles_x, tiles_y,
            target, accumulate);
    }
    NSS_CUDA_CHECK_LAUNCH();
}

void aggregate_atomic(const float* values, const AggregatePatch* patches, int npatch, int block,
                      const AggregateTarget& target, cudaStream_t stream) {
    const long long total = static_cast<long long>(npatch) * block * block;
    if (total == 0) return;
    atomic_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(values, patches, total, block, target);
    NSS_CUDA_CHECK_LAUNCH();
}

void aggregate_finish(const float* num, const float* den, const float* src, int width, int height, int pitch,
                      float* out, cudaStream_t stream) {
    const dim3 threads(32, 8);
    const dim3 blocks((width + 31) / 32, (height + 7) / 8);
    finish_kernel<<<blocks, threads, 0, stream>>>(num, den, src, width, height, pitch, out);
    NSS_CUDA_CHECK_LAUNCH();
}

void accumulate_slice(float* acc_num, float* acc_den, const float* num, const float* den, std::size_t count,
                      cudaStream_t stream) {
    if (count == 0) return;
    accumulate_slice_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(acc_num, acc_den, num, den,
                                                                                         count);
    NSS_CUDA_CHECK_LAUNCH();
}

}  // namespace nss_cuda
