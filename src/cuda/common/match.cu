// SPDX-License-Identifier: GPL-2.0-only
// Block matching kernels (see match.hpp for the contract). One thread block
// per reference patch; candidates stream through the deterministic top-K.
// SSDs are summed row-major in the same order on every path, so the staged
// (shared-memory window) and global variants return identical matches.
#include "cuda/common/match.hpp"
#include "cuda/common/topk.cuh"
#include "cuda/runtime/error.hpp"

#include <algorithm>

namespace nss_cuda {
namespace {

using detail::BlockTopK;
using detail::MatchKey;

constexpr int kThreads = 128;
constexpr int kTemporalSort = 1024;
constexpr std::size_t kSharedBudget = 48 * 1024;

__device__ __forceinline__ float block_ssd(const float* a, int a_pitch, const float* b, int b_pitch, int block) {
    float sum = 0.f;
    for (int r = 0; r < block; ++r) {
        const float* ra = a + r * a_pitch;
        const float* rb = b + r * b_pitch;
        for (int c = 0; c < block; ++c) {
            const float d = ra[c] - rb[c];
            sum = fmaf(d, d, sum);
        }
    }
    return sum;
}

struct Window {
    int left, top, right, bottom;  // candidate position range (inclusive)
};

__device__ __forceinline__ Window spatial_window(const MatchGeometry& g, int cx, int cy) {
    const int range = max(g.bm_range, 0);
    return Window{max(cx - range, 0), max(cy - range, 0), min(cx + range, g.width - g.block),
                  min(cy + range, g.height - g.block)};
}

__device__ void write_result(DeviceMatch* dst, int* count, int cx, int cy, int t0, const MatchKey* keys, int n) {
    for (int i = threadIdx.x; i <= n; i += blockDim.x) {
        if (i == 0) {
            dst[0] = DeviceMatch{cx, cy, t0, 0.f};
        } else {
            const MatchKey& k = keys[i - 1];
            dst[i] = DeviceMatch{detail::key_x(k), detail::key_y(k), detail::key_t(k), detail::key_distance(k)};
        }
    }
    if (threadIdx.x == 0) *count = 1 + n;
}

// Scans the spatial window around (cx, cy) into topk. With a staged window
// the pixels come from shared memory, otherwise from global memory; the
// arithmetic is identical.
template <int N>
__device__ void scan_spatial(BlockTopK<N>& topk, const float* anchor, int anchor_pitch, const float* base,
                             int base_pitch, int base_x, int base_y, const MatchGeometry& g, const Window& w, int cx,
                             int cy, int t) {
    const int ww = w.right - w.left + 1;
    const int total = ww * (w.bottom - w.top + 1);
    for (int start = 0; start < total; start += topk.chunk) {
        for (int i = threadIdx.x; i < topk.chunk; i += blockDim.x) {
            const int idx = start + i;
            MatchKey key = detail::sentinel_key();
            if (idx < total) {
                const int x = w.left + idx % ww;
                const int y = w.top + idx / ww;
                if (x != cx || y != cy) {
                    const float* cand = base + static_cast<long long>(y - base_y) * base_pitch + (x - base_x);
                    key = detail::make_key(block_ssd(anchor, anchor_pitch, cand, base_pitch, g.block), t, y, x);
                }
            }
            topk.candidate(i) = key;
        }
        __syncthreads();
        topk.merge();
    }
}

template <int N>
__global__ void __launch_bounds__(kThreads) spatial_match_kernel(const float* plane, MatchGeometry g, RasterGrid grid,
                                                                 int ref_begin, DeviceMatch* out, int* counts,
                                                                 bool staged) {
    __shared__ MatchKey keys[N];
    extern __shared__ float window[];
    const int ref = blockIdx.x;
    const int cx = grid.x(ref_begin + ref);
    const int cy = grid.y(ref_begin + ref);
    DeviceMatch* dst = out + static_cast<long long>(ref) * g.group;
    const int wanted = min(g.group, kMaxGroup);
    if (wanted <= 1) {
        write_result(dst, counts + ref, cx, cy, 0, keys, 0);
        return;
    }
    const Window w = spatial_window(g, cx, cy);
    BlockTopK<N> topk;
    topk.init(keys, wanted - 1);
    if (staged) {
        const int sw = w.right - w.left + g.block;
        const int sh = w.bottom - w.top + g.block;
        for (int i = threadIdx.x; i < sw * sh; i += blockDim.x) {
            window[i] = plane[static_cast<long long>(w.top + i / sw) * g.pitch + w.left + i % sw];
        }
        __syncthreads();
        const float* anchor = window + (cy - w.top) * sw + (cx - w.left);
        scan_spatial(topk, anchor, sw, window, sw, w.left, w.top, g, w, cx, cy, 0);
    } else {
        const float* anchor = plane + static_cast<long long>(cy) * g.pitch + cx;
        scan_spatial(topk, anchor, g.pitch, plane, g.pitch, 0, 0, g, w, cx, cy, 0);
    }
    write_result(dst, counts + ref, cx, cy, 0, keys, topk.filled);
}

// Warp-per-reference variant for small top-K (the common BM3D/WNNM case):
// each warp stages its search window and candidate keys in its own shared
// memory, then extracts the K smallest keys by K warp-wide argmin rounds.
// The keys are a total order, so the result equals the block top-K path.
constexpr int kWarpRefs = kThreads / 32;
constexpr int kWarpMaxK = 16;

__device__ __forceinline__ void warp_min(MatchKey& key, int& index) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        MatchKey other{__shfl_xor_sync(0xffffffffu, key.hi, offset), __shfl_xor_sync(0xffffffffu, key.lo, offset)};
        const int other_index = __shfl_xor_sync(0xffffffffu, index, offset);
        if (detail::key_less(other, key)) {
            key = other;
            index = other_index;
        }
    }
}

__global__ void __launch_bounds__(kThreads) spatial_match_warp_kernel(const float* plane, MatchGeometry g,
                                                                      RasterGrid grid, int ref_begin, int ref_count,
                                                                      DeviceMatch* out, int* counts, int window_floats,
                                                                      int key_capacity) {
    extern __shared__ unsigned char shared[];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int ref = blockIdx.x * kWarpRefs + warp;
    if (ref >= ref_count) return;  // warp-uniform; no block barriers below
    auto* keys = reinterpret_cast<MatchKey*>(shared) + warp * key_capacity;
    float* window = reinterpret_cast<float*>(reinterpret_cast<MatchKey*>(shared) + kWarpRefs * key_capacity) +
                    warp * window_floats;
    const int cx = grid.x(ref_begin + ref);
    const int cy = grid.y(ref_begin + ref);
    DeviceMatch* dst = out + static_cast<long long>(ref) * g.group;
    const int wanted = min(g.group, kMaxGroup) - 1;
    const Window w = spatial_window(g, cx, cy);
    const int sw = w.right - w.left + g.block;
    const int sh = w.bottom - w.top + g.block;
    for (int i = lane; i < sw * sh; i += 32) {
        window[i] = plane[static_cast<long long>(w.top + i / sw) * g.pitch + w.left + i % sw];
    }
    __syncwarp();
    const int ww = w.right - w.left + 1;
    const int total = ww * (w.bottom - w.top + 1);
    const float* anchor = window + (cy - w.top) * sw + (cx - w.left);
    for (int idx = lane; idx < total; idx += 32) {
        const int x = w.left + idx % ww;
        const int y = w.top + idx / ww;
        keys[idx] = (x == cx && y == cy)
                        ? detail::sentinel_key()
                        : detail::make_key(block_ssd(anchor, sw, window + (y - w.top) * sw + (x - w.left), sw, g.block),
                                           0, y, x);
    }
    __syncwarp();
    int filled = 0;
    for (; filled < wanted; ++filled) {
        MatchKey best = detail::sentinel_key();
        int index = -1;
        for (int idx = lane; idx < total; idx += 32) {
            if (detail::key_less(keys[idx], best)) {
                best = keys[idx];
                index = idx;
            }
        }
        warp_min(best, index);
        if (index < 0) break;  // fewer candidates than wanted
        if (lane == 0) {
            dst[1 + filled] =
                DeviceMatch{detail::key_x(best), detail::key_y(best), detail::key_t(best), detail::key_distance(best)};
            keys[index] = detail::sentinel_key();
        }
        __syncwarp();
    }
    if (lane == 0) {
        dst[0] = DeviceMatch{cx, cy, 0, 0.f};
        counts[ref] = 1 + filled;
    }
}

struct Center {
    int x;
    int y;
};

__global__ void __launch_bounds__(kThreads) predictive_match_kernel(MatchGeometry g, TemporalWindow tw,
                                                                    RasterGrid grid, int ref_begin, DeviceMatch* out,
                                                                    int* counts) {
    constexpr int N = kTemporalSort;
    __shared__ MatchKey keys[N];
    __shared__ MatchKey global_keys[128];
    __shared__ Center seeds[kMaxGroup];
    __shared__ Center centers[kMaxGroup];
    __shared__ int global_count;
    __shared__ int center_count;

    const int ref = blockIdx.x;
    const int cx = grid.x(ref_begin + ref);
    const int cy = grid.y(ref_begin + ref);
    const int t0 = tw.t0;
    DeviceMatch* dst = out + static_cast<long long>(ref) * g.group;
    const int wanted = min(g.group, kMaxGroup);
    if (wanted <= 1) {
        write_result(dst, counts + ref, cx, cy, t0, keys, 0);
        return;
    }
    const float* anchor_plane = tw.frames[t0];
    const float* anchor = anchor_plane + static_cast<long long>(cy) * g.pitch + cx;
    BlockTopK<N> topk;
    topk.init(keys, wanted - 1);
    scan_spatial(topk, anchor, g.pitch, anchor_plane, g.pitch, 0, 0, g, spatial_window(g, cx, cy), cx, cy, t0);
    const int spatial = topk.filled;
    if (tw.radius == 0 || tw.ntemp == 1) {
        write_result(dst, counts + ref, cx, cy, t0, keys, spatial);
        return;
    }

    // Seeds: the reference itself followed by the best spatial matches.
    const int seed_count = min(min(1 + spatial, tw.ps_num), kMaxGroup);
    for (int i = threadIdx.x; i < 128; i += blockDim.x) {
        global_keys[i] = i < spatial ? keys[i] : detail::sentinel_key();
    }
    for (int i = threadIdx.x; i < seed_count; i += blockDim.x) {
        seeds[i] = i == 0 ? Center{cx, cy} : Center{detail::key_x(keys[i - 1]), detail::key_y(keys[i - 1])};
    }
    if (threadIdx.x == 0) global_count = spatial;
    __syncthreads();

    const int first = max(0, tw.valid_begin);
    const int last = tw.valid_end < 0 ? tw.ntemp : min(tw.ntemp, tw.valid_end);
    const int local_k = min(tw.ps_num, g.group);
    const int pr = tw.ps_range;
    const int max_x = g.width - g.block;
    const int max_y = g.height - g.block;

    for (int sign = -1; sign <= 1; sign += 2) {
        for (int i = threadIdx.x; i < seed_count; i += blockDim.x) centers[i] = seeds[i];
        if (threadIdx.x == 0) center_count = seed_count;
        __syncthreads();
        for (int dt = 1; dt <= tw.radius; ++dt) {
            const int t = t0 + sign * dt;
            if (t < first || t >= last) break;
            const int nc = center_count;
            int ymin = g.height, ymax = -1, xmin = g.width, xmax = -1;
            for (int i = 0; i < nc; ++i) {
                ymin = min(ymin, max(0, centers[i].y - pr));
                ymax = max(ymax, min(max_y, centers[i].y + pr));
                xmin = min(xmin, max(0, centers[i].x - pr));
                xmax = max(xmax, min(max_x, centers[i].x + pr));
            }
            const float* plane = tw.frames[t];
            topk.init(keys, local_k);
            if (ymax >= ymin && xmax >= xmin) {
                const int bw = xmax - xmin + 1;
                const int total = bw * (ymax - ymin + 1);
                for (int start = 0; start < total; start += topk.chunk) {
                    for (int i = threadIdx.x; i < topk.chunk; i += blockDim.x) {
                        const int idx = start + i;
                        MatchKey key = detail::sentinel_key();
                        if (idx < total) {
                            const int x = xmin + idx % bw;
                            const int y = ymin + idx / bw;
                            bool inside = false;
                            for (int c = 0; c < nc && !inside; ++c) {
                                inside = abs(y - centers[c].y) <= pr && abs(x - centers[c].x) <= pr;
                            }
                            if (inside) {
                                const float* cand = plane + static_cast<long long>(y) * g.pitch + x;
                                key = detail::make_key(block_ssd(anchor, g.pitch, cand, g.pitch, g.block), t, y, x);
                            }
                        }
                        topk.candidate(i) = key;
                    }
                    __syncthreads();
                    topk.merge();
                }
            }
            const int local = topk.filled;
            // Global merge: current global (<= 63) + this layer's local (<= 64).
            const int gc = global_count;
            for (int i = threadIdx.x; i < 128; i += blockDim.x) {
                if (i >= gc && i < 64) global_keys[i] = detail::sentinel_key();
                if (i >= 64) global_keys[i] = i - 64 < local ? keys[i - 64] : detail::sentinel_key();
            }
            __syncthreads();
            detail::bitonic_sort<128>(global_keys);
            if (threadIdx.x == 0) {
                int valid = 0;
                while (valid < wanted - 1 &&
                       (global_keys[valid].hi != ~0ull || global_keys[valid].lo != ~0ull)) {
                    ++valid;
                }
                global_count = valid;
                center_count = local;
            }
            for (int i = threadIdx.x; i < local; i += blockDim.x) {
                centers[i] = Center{detail::key_x(keys[i]), detail::key_y(keys[i])};
            }
            __syncthreads();
            if (local == 0) break;
        }
    }
    write_result(dst, counts + ref, cx, cy, t0, global_keys, global_count);
}

template <int N>
void launch_spatial(const float* plane, const MatchGeometry& g, const RasterGrid& grid, int ref_begin, int ref_count,
                    DeviceMatch* out, int* counts, cudaStream_t stream) {
    const int range = std::max(g.bm_range, 0);
    const std::size_t window = static_cast<std::size_t>(2 * range + g.block) * (2 * range + g.block) * sizeof(float);
    const bool staged = window + N * sizeof(MatchKey) <= kSharedBudget;
    spatial_match_kernel<N><<<ref_count, kThreads, staged ? window : 0, stream>>>(plane, g, grid, ref_begin, out,
                                                                                    counts, staged);
}

}  // namespace

RasterGrid make_raster_grid(int width, int height, int block, int step) {
    RasterGrid grid{0, 0, step, width - block, height - block};
    if (width < block || height < block || step < 1) return grid;
    grid.nx = (width - block + step - 1) / step + 1;
    grid.ny = (height - block + step - 1) / step + 1;
    return grid;
}

void spatial_match(const float* plane, const MatchGeometry& geometry, const RasterGrid& grid, int ref_begin,
                   int ref_count, DeviceMatch* out, int* counts, cudaStream_t stream) {
    if (ref_count <= 0) return;
    const int range = std::max(geometry.bm_range, 0);
    const int candidates = (2 * range + 1) * (2 * range + 1);
    const int wanted = std::min(geometry.group, kMaxGroup) - 1;
    const int window_floats = (2 * range + geometry.block) * (2 * range + geometry.block);
    const std::size_t warp_bytes =
        kWarpRefs * (static_cast<std::size_t>(candidates) * sizeof(MatchKey) + window_floats * sizeof(float));
    if (wanted >= 1 && wanted <= kWarpMaxK && warp_bytes <= kSharedBudget) {
        spatial_match_warp_kernel<<<(ref_count + kWarpRefs - 1) / kWarpRefs, kThreads, warp_bytes, stream>>>(
            plane, geometry, grid, ref_begin, ref_count, out, counts, window_floats, candidates);
        NSS_CUDA_CHECK_LAUNCH();
        return;
    }
    // Block top-K: smallest sort that takes the whole (2r+1)^2 window in one chunk.
    const int front = detail::pow2_at_least(std::max(wanted, 1));
    if (front + candidates <= 256) {
        launch_spatial<256>(plane, geometry, grid, ref_begin, ref_count, out, counts, stream);
    } else if (front + candidates <= 512) {
        launch_spatial<512>(plane, geometry, grid, ref_begin, ref_count, out, counts, stream);
    } else {
        launch_spatial<1024>(plane, geometry, grid, ref_begin, ref_count, out, counts, stream);
    }
    NSS_CUDA_CHECK_LAUNCH();
}

void predictive_match(const MatchGeometry& geometry, const TemporalWindow& window, const RasterGrid& grid,
                      int ref_begin, int ref_count, DeviceMatch* out, int* counts, cudaStream_t stream) {
    if (ref_count <= 0) return;
    predictive_match_kernel<<<ref_count, kThreads, 0, stream>>>(geometry, window, grid, ref_begin, out, counts);
    NSS_CUDA_CHECK_LAUNCH();
}

}  // namespace nss_cuda
