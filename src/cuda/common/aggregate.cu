// SPDX-License-Identifier: GPL-2.0-only
#include "cuda/common/aggregate.hpp"
#include "cuda/runtime/error.hpp"

#include <cub/device/device_radix_sort.cuh>

#include <algorithm>
#include <stdexcept>

namespace nss_cuda {
namespace {

// Tiles are 16x16 for blocks up to 16 (one pixel per thread) and 32x32 above;
// with fine tiles, 8x8 for blocks up to 8, and with micro tiles 4x4 for
// blocks up to 4. A block <= tile + 1 spans at most
// 2x2 tiles, and the smaller the tile, the fewer entries of its bin miss a
// given pixel. The per-pixel order (patch id) does not depend on the tile
// size.
constexpr int kMicroTile = 4;
constexpr int kTinyTile = 8;
constexpr int kSmallTile = 16;
constexpr int kLargeTile = OrderedAggregator::kTile;
constexpr int kTilesPerPatch = 4;
__host__ __device__ constexpr int tile_for(int block, bool fine, bool micro) {
    return micro && block <= kMicroTile ? kMicroTile
           : fine && block <= kTinyTile ? kTinyTile
           : block <= kSmallTile        ? kSmallTile
                                        : kLargeTile;
}
constexpr int kTileThreads = 256;
template <int kTile>
constexpr int kTileBlock = kTile * kTile < kTileThreads ? kTile * kTile : kTileThreads;

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
                   const int* end, int block, int tiles_x, int tiles_y, int tile_y0, AggregateTarget target,
                   bool accumulate, const float* pixel_den, int den_group) {
    constexpr int kThreads = kTileBlock<kTile>;
    constexpr int kPerThread = kTile * kTile / kThreads;
    constexpr int kRowStep = kThreads / kTile;
    __shared__ AggregatePatch staged[kThreads];
    __shared__ int staged_id[kThreads];
    const int tx = blockIdx.x, ty = tile_y0 + static_cast<int>(blockIdx.y), slice = blockIdx.z;
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
    for (int chunk = first; chunk < last; chunk += kThreads) {
        const int n = min(kThreads, last - chunk);
        __syncthreads();
        if (threadIdx.x < n) {
            const int p = ids[chunk + threadIdx.x];
            staged_id[threadIdx.x] = p;
            staged[threadIdx.x] = patches[p];
        }
        __syncthreads();
        if constexpr (kPerThread == 1) {
            // One pixel per thread: the sum is a chain of dependent FMAs whose
            // operands come from global memory. The loads of kAhead entries
            // are issued together, ahead of their additions (same order, same
            // sums); an entry that misses the pixel loads nothing.
            constexpr int kAhead = 4;
            const int py = ty * kTile + row0;
            float sum_num = num[0], sum_den = den[0];
            for (int e = 0; e < n; e += kAhead) {
                float weight[kAhead], value[kAhead], add[kAhead];
                bool hit[kAhead];
#pragma unroll
                for (int u = 0; u < kAhead; ++u) {
                    const int at = min(e + u, n - 1);
                    const AggregatePatch patch = staged[at];
                    const int dx = px - patch.x, dy = py - patch.y;
                    hit[u] = e + u < n && dx >= 0 && dx < block && dy >= 0 && dy < block;
                    const long long cell = static_cast<long long>(staged_id[at]) * area + dy * block + dx;
                    weight[u] = patch.weight;
                    value[u] = hit[u] ? values[cell] : 0.f;
                    add[u] = !hit[u] ? 0.f
                             : pixel_den ? pixel_den[static_cast<long long>(staged_id[at] / den_group) * area + dy * block + dx]
                                         : patch.weight;
                }
#pragma unroll
                for (int u = 0; u < kAhead; ++u) {
                    if (hit[u]) {
                        sum_num = fmaf(weight[u], value[u], sum_num);
                        sum_den += add[u];
                    }
                }
            }
            num[0] = sum_num;
            den[0] = sum_den;
        } else {
            for (int e = 0; e < n; ++e) {
                const AggregatePatch patch = staged[e];
                const int dx = px - patch.x;
                if (dx < 0 || dx >= block) continue;
                const float* v = values + static_cast<long long>(staged_id[e]) * area + dx;
                const float* w =
                    pixel_den ? pixel_den + static_cast<long long>(staged_id[e] / den_group) * area + dx : nullptr;
#pragma unroll
                for (int k = 0; k < kPerThread; ++k) {
                    const int dy = ty * kTile + row0 + k * kRowStep - patch.y;
                    if (dy >= 0 && dy < block) {
                        num[k] = fmaf(patch.weight, v[dy * block], num[k]);
                        den[k] += w ? w[dy * block] : patch.weight;
                    }
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

// tile_reduce_kernel for 8x8 tiles with many entries each (and patch weights
// as the denominator's terms). There a tile's block is one warp's chain of
// some 26 instructions and a wait for global memory per four entries, and
// the launch waits for the fullest tile. Here the work on a chunk of entries
// is three steps of a pipeline, each on threads of its own: 32 threads bring
// the patch records to shared memory; 384, six to a pixel, gather the value
// and the weight of every (entry, pixel), zeros where the patch misses the
// pixel, a thread's loads all at once; the 64 pixel threads add them in
// entry order (the same additions: a miss adds zeros), about five
// instructions an entry. Same sums as tile_reduce_kernel, bit for bit.
constexpr int kStreamGather = 6;  // threads to a pixel
template <int kTile>
constexpr int kStreamPixels = kTile * kTile;
// Entries a step: one per staging thread, at most 32.
template <int kTile>
constexpr int kStreamChunk = kStreamPixels<kTile> < 32 ? kStreamPixels<kTile> : 32;
template <int kTile>
constexpr int kStreamThreads = (2 + kStreamGather) * kStreamPixels<kTile>;

template <int kTile>
__global__ void __launch_bounds__(kStreamThreads<kTile>)
tile_stream_kernel(const float* values, const AggregatePatch* patches, const int* ids, const int* begin,
                   const int* end, int block, int tiles_x, int tiles_y, int tile_y0, AggregateTarget target,
                   bool accumulate) {
    constexpr int kPixels = kStreamPixels<kTile>;
    constexpr int kChunk = kStreamChunk<kTile>;
    __shared__ AggregatePatch staged[2][kChunk];
    __shared__ int staged_id[2][kChunk];
    __shared__ float staged_value[2][kChunk * kPixels];
    __shared__ float staged_weight[2][kChunk * kPixels];
    const int tx = blockIdx.x, ty = tile_y0 + static_cast<int>(blockIdx.y), slice = blockIdx.z;
    const int bin = slice * tiles_x * tiles_y + ty * tiles_x + tx;
    // Threads [0, pixels) add, [pixels, 2 pixels) stage, the others gather.
    const int role = threadIdx.x / kPixels;
    const int pixel = threadIdx.x % kPixels;
    const int px = tx * kTile + pixel % kTile;
    const int py = ty * kTile + pixel / kTile;
    const bool inside = px < target.width && py < target.height;
    float* tnum = target.num + slice * target.slice_step + static_cast<long long>(py) * target.pitch + px;
    float* tden = target.den + slice * target.slice_step + static_cast<long long>(py) * target.pitch + px;
    float num = 0.f, den = 0.f;
    if (role == 0 && inside && accumulate) num = *tnum, den = *tden;
    const int area = block * block;
    const int first = begin[bin], last = end[bin];
    const int steps = (last - first + kChunk - 1) / kChunk;
    const auto count = [=](int chunk) { return min(kChunk, last - first - chunk * kChunk); };
    // Step s: chunk s + 2 is staged, chunk s + 1 gathered, chunk s added.
    for (int s = -2; s < steps; ++s) {
        if (role == 1) {
            if (s + 2 < steps && pixel < count(s + 2)) {
                const int id = ids[first + (s + 2) * kChunk + pixel];
                staged_id[s & 1][pixel] = id;
                staged[s & 1][pixel] = patches[id];
            }
        } else if (role >= 2) {
            if (s + 1 >= 0 && s + 1 < steps) {
                const int n = count(s + 1);
                const AggregatePatch* patch = staged[(s + 1) & 1];
                const int* id = staged_id[(s + 1) & 1];
                float* value = staged_value[(s + 1) & 1];
                float* weight = staged_weight[(s + 1) & 1];
                // A thread's entries: role - 2, then every kStreamGather-th.
                constexpr int kLoads = (kChunk + kStreamGather - 1) / kStreamGather;
                float v[kLoads], w[kLoads];
#pragma unroll
                for (int u = 0; u < kLoads; ++u) {
                    const int at = min(role - 2 + kStreamGather * u, n - 1);
                    const int dx = px - patch[at].x, dy = py - patch[at].y;
                    const bool hit = dx >= 0 && dx < block && dy >= 0 && dy < block;
                    v[u] = hit ? values[static_cast<long long>(id[at]) * area + dy * block + dx] : 0.f;
                    w[u] = hit ? patch[at].weight : 0.f;
                }
#pragma unroll
                for (int u = 0; u < kLoads; ++u) {
                    const int e = role - 2 + kStreamGather * u;
                    if (e < n) {
                        value[e * kPixels + pixel] = v[u];
                        weight[e * kPixels + pixel] = w[u];
                    }
                }
            }
        } else if (s >= 0) {
            const int n = count(s);
            const float* value = staged_value[s & 1];
            const float* weight = staged_weight[s & 1];
            int e = 0;
            for (; e + 4 <= n; e += 4) {
                float v[4], w[4];
#pragma unroll
                for (int u = 0; u < 4; ++u) v[u] = value[(e + u) * kPixels + pixel], w[u] = weight[(e + u) * kPixels + pixel];
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    num = fmaf(w[u], v[u], num);
                    den += w[u];
                }
            }
            for (; e < n; ++e) {
                const float w = weight[e * kPixels + pixel];
                num = fmaf(w, value[e * kPixels + pixel], num);
                den += w;
            }
        }
        __syncthreads();
    }
    if (role == 0 && inside) {
        *tnum = num;
        *tden = den;
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

// rows[y][x]: the weights of the patches of row y that cover column x.
__global__ void fixed_rows_kernel(const unsigned long long* corner, int block, int width, int height, int pitch,
                                  unsigned long long* rows) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const unsigned long long* at = corner + static_cast<long long>(y) * pitch + x;
    unsigned long long sum = 0;
    for (int j = 0; j < block && j <= x; ++j) sum += at[-j];
    rows[static_cast<long long>(y) * pitch + x] = sum;
}

__global__ void fixed_finish_kernel(const unsigned long long* fixed_num, const unsigned long long* rows, int block,
                                    const float* src, int width, int height, int pitch, float* out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const long long i = static_cast<long long>(y) * pitch + x;
    unsigned long long weight = 0;
    for (int k = 0; k < block && k <= y; ++k) weight += rows[i - static_cast<long long>(k) * pitch];
    constexpr float kUnit = 1.f / 4294967296.f;  // 1 / kFixedScale
    const float num = static_cast<float>(static_cast<long long>(fixed_num[i])) * kUnit;
    const float den = static_cast<float>(static_cast<long long>(weight)) * kUnit;
    out[i] = den > 1e-12f ? num / den : src[i];
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

__global__ void accumulate_identity_kernel(float* acc_num, float* acc_den, const float* src, std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    acc_num[i] += src[i];
    acc_den[i] += 1.f;
}

__global__ void regularize_kernel(float* x, const float* y, std::size_t count, float delta) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    x[i] = fmaf(delta, y[i] - x[i], x[i]);
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

std::size_t OrderedAggregator::bin_bytes(int max_width, int max_height, int max_slices, bool fine_tiles,
                                         bool micro_tiles) {
    const int tile = micro_tiles ? kMicroTile : fine_tiles ? kTinyTile : kSmallTile;
    return 2 * sizeof(int) * (static_cast<std::size_t>((max_width + tile - 1) / tile) *
                              ((max_height + tile - 1) / tile) * max_slices);
}

OrderedAggregator::OrderedAggregator(int max_width, int max_height, int max_slices, std::size_t max_patches,
                                     const std::shared_ptr<nss::ResourceBudget>& budget, bool fine_tiles,
                                     bool micro_tiles)
    : fine_tiles_(fine_tiles), micro_tiles_(micro_tiles),
      max_bins_(bin_bytes(max_width, max_height, max_slices, fine_tiles, micro_tiles) / (2 * sizeof(int))),
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
                            const AggregateTarget& target, cudaStream_t stream, bool accumulate,
                            const float* pixel_den, int den_group, int row_first, int row_last) {
    const int tile = tile_for(block, fine_tiles_, micro_tiles_);
    const int tiles_x = (target.width + tile - 1) / tile;
    const int tiles_y = (target.height + tile - 1) / tile;
    const std::size_t bins = static_cast<std::size_t>(tiles_x) * tiles_y * target.slices;
    if (static_cast<std::size_t>(npatch) > max_patches_ || bins > max_bins_ || block < 1 || block > kMaxBlock) {
        throw std::logic_error("nss_cuda: aggregation over capacity");
    }
    // With a band, only its tile rows are binned and run: a batch's patches
    // cover a band of the frame, and the density test below is the band's.
    // Pixels outside it keep their sums, so an overwrite clears the target
    // first and then adds (the same sums: they start from zero either way).
    const bool band = row_first > 0 || (row_last >= 0 && row_last < target.height - 1);
    const int tile_y0 = band ? row_first / tile : 0;
    const int tile_y1 = band ? std::min(row_last / tile, tiles_y - 1) : tiles_y - 1;
    if (band && !accumulate) {
        for (int slice = 0; slice < target.slices; ++slice) {
            NSS_CUDA_CHECK(cudaMemset2DAsync(target.num + slice * target.slice_step, target.pitch * sizeof(float), 0,
                                             target.width * sizeof(float), target.height, stream));
            NSS_CUDA_CHECK(cudaMemset2DAsync(target.den + slice * target.slice_step, target.pitch * sizeof(float), 0,
                                             target.width * sizeof(float), target.height, stream));
        }
        accumulate = true;
    }
    // Bins the kernels read: the band's rows of every slice (in between, rows
    // no key reaches).
    const std::size_t bin_first = static_cast<std::size_t>(tile_y0) * tiles_x;
    const std::size_t bin_count =
        (static_cast<std::size_t>(target.slices - 1) * tiles_y + tile_y1 + 1) * tiles_x - bin_first;
    NSS_CUDA_CHECK(cudaMemsetAsync(bin_begin_.as<int>() + bin_first, 0, bin_count * sizeof(int), stream));
    NSS_CUDA_CHECK(cudaMemsetAsync(bin_end_.as<int>() + bin_first, 0, bin_count * sizeof(int), stream));
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
    const int band_rows = tile_y1 - tile_y0 + 1;
    const dim3 grid(tiles_x, band_rows, target.slices);
    // Same sums from either kernel; the stream kernel's larger blocks pay
    // where the tiles hold several chunks on average.
    const std::size_t run_bins = static_cast<std::size_t>(tiles_x) * band_rows * target.slices;
    const bool dense = !pixel_den && tile <= kTinyTile &&
                       static_cast<std::size_t>(entries) >=
                           4 * static_cast<std::size_t>(tile == kMicroTile ? kStreamChunk<kMicroTile> : kStreamChunk<kTinyTile>) *
                               run_bins;
    if (dense && tile == kMicroTile) {
        tile_stream_kernel<kMicroTile><<<grid, kStreamThreads<kMicroTile>, 0, stream>>>(
            values, patches, ids_out_.as<int>(), bin_begin_.as<int>(), bin_end_.as<int>(), block, tiles_x, tiles_y,
            tile_y0, target, accumulate);
    } else if (dense) {
        tile_stream_kernel<kTinyTile><<<grid, kStreamThreads<kTinyTile>, 0, stream>>>(
            values, patches, ids_out_.as<int>(), bin_begin_.as<int>(), bin_end_.as<int>(), block, tiles_x, tiles_y,
            tile_y0, target, accumulate);
    } else if (tile == kMicroTile) {
        tile_reduce_kernel<kMicroTile><<<grid, kTileBlock<kMicroTile>, 0, stream>>>(
            values, patches, ids_out_.as<int>(), bin_begin_.as<int>(), bin_end_.as<int>(), block, tiles_x, tiles_y,
            tile_y0, target, accumulate, pixel_den, den_group);
    } else if (tile == kTinyTile) {
        tile_reduce_kernel<kTinyTile><<<grid, kTileBlock<kTinyTile>, 0, stream>>>(
            values, patches, ids_out_.as<int>(), bin_begin_.as<int>(), bin_end_.as<int>(), block, tiles_x, tiles_y,
            tile_y0, target, accumulate, pixel_den, den_group);
    } else if (tile == kSmallTile) {
        tile_reduce_kernel<kSmallTile><<<grid, kTileThreads, 0, stream>>>(
            values, patches, ids_out_.as<int>(), bin_begin_.as<int>(), bin_end_.as<int>(), block, tiles_x, tiles_y,
            tile_y0, target, accumulate, pixel_den, den_group);
    } else {
        tile_reduce_kernel<kLargeTile><<<grid, kTileThreads, 0, stream>>>(
            values, patches, ids_out_.as<int>(), bin_begin_.as<int>(), bin_end_.as<int>(), block, tiles_x, tiles_y,
            tile_y0, target, accumulate, pixel_den, den_group);
    }
    NSS_CUDA_CHECK_LAUNCH();
}

void fixed_clear(const FixedTarget& target, std::size_t count, cudaStream_t stream) {
    NSS_CUDA_CHECK(cudaMemsetAsync(target.num, 0, count * sizeof(unsigned long long), stream));
    NSS_CUDA_CHECK(cudaMemsetAsync(target.den, 0, count * sizeof(unsigned long long), stream));
}

void fixed_finish(const FixedTarget& target, int block, const float* src, int width, int height,
                  unsigned long long* rows, float* out, cudaStream_t stream) {
    const dim3 threads(32, 8);
    const dim3 blocks((width + 31) / 32, (height + 7) / 8);
    fixed_rows_kernel<<<blocks, threads, 0, stream>>>(target.den, block, width, height, target.pitch, rows);
    NSS_CUDA_CHECK_LAUNCH();
    fixed_finish_kernel<<<blocks, threads, 0, stream>>>(target.num, rows, block, src, width, height, target.pitch, out);
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

void accumulate_identity(float* acc_num, float* acc_den, const float* src, std::size_t count, cudaStream_t stream) {
    if (count == 0) return;
    accumulate_identity_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(acc_num, acc_den, src,
                                                                                            count);
    NSS_CUDA_CHECK_LAUNCH();
}

void iter_regularize(float* x, const float* y, std::size_t count, float delta, cudaStream_t stream) {
    if (count == 0) return;
    regularize_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(x, y, count, delta);
    NSS_CUDA_CHECK_LAUNCH();
}

}  // namespace nss_cuda
