// SPDX-License-Identifier: GPL-2.0-only
// Block matching kernels (see match.hpp for the contract). One thread block
// per reference patch; candidates stream through the deterministic top-K.
#include "cuda/common/match.hpp"
#include "cuda/common/topk.cuh"
#include "cuda/runtime/error.hpp"

#include <algorithm>

namespace nss_cuda {
namespace {

using detail::BlockTopK;
using detail::MatchKey;

constexpr int kSortSize = 1024;
constexpr int kThreads = 256;

__device__ __forceinline__ float block_ssd(const float* a, const float* b, int pitch, int block) {
    float sum = 0.f;
    for (int r = 0; r < block; ++r) {
        const float* ra = a + r * pitch;
        const float* rb = b + r * pitch;
        for (int c = 0; c < block; ++c) {
            const float d = ra[c] - rb[c];
            sum = fmaf(d, d, sum);
        }
    }
    return sum;
}

// Spatial window scan at frame t into topk; the centre (cx, cy) is excluded.
__device__ void scan_spatial(BlockTopK<kSortSize>& topk, const float* ref_plane, const float* plane,
                             const MatchGeometry& g, int cx, int cy, int t) {
    const int max_x = g.width - g.block;
    const int max_y = g.height - g.block;
    const int range = max(g.bm_range, 0);
    const int top = max(cy - range, 0);
    const int bottom = min(cy + range, max_y);
    const int left = max(cx - range, 0);
    const int right = min(cx + range, max_x);
    const int w = right - left + 1;
    const int total = w * (bottom - top + 1);
    const float* anchor = ref_plane + cy * g.pitch + cx;
    for (int base = 0; base < total; base += BlockTopK<kSortSize>::kChunk) {
        for (int i = threadIdx.x; i < BlockTopK<kSortSize>::kChunk; i += blockDim.x) {
            const int idx = base + i;
            MatchKey key = detail::sentinel_key();
            if (idx < total) {
                const int x = left + idx % w;
                const int y = top + idx / w;
                if (x != cx || y != cy) {
                    key = detail::make_key(block_ssd(anchor, plane + y * g.pitch + x, g.pitch, g.block), t, y, x);
                }
            }
            topk.candidate(i) = key;
        }
        __syncthreads();
        topk.merge();
    }
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

__global__ void __launch_bounds__(kThreads) spatial_match_kernel(const float* plane, MatchGeometry g, RasterGrid grid,
                                                                 DeviceMatch* out, int* counts) {
    __shared__ MatchKey keys[kSortSize];
    const int ref = blockIdx.x;
    const int cx = grid.x(ref);
    const int cy = grid.y(ref);
    DeviceMatch* dst = out + static_cast<long long>(ref) * kMaxGroup;
    const int wanted = min(g.group, kMaxGroup);
    if (wanted <= 1) {
        write_result(dst, counts + ref, cx, cy, 0, keys, 0);
        return;
    }
    BlockTopK<kSortSize> topk;
    topk.init(keys, wanted - 1);
    scan_spatial(topk, plane, plane, g, cx, cy, 0);
    write_result(dst, counts + ref, cx, cy, 0, keys, topk.filled);
}

struct Center {
    int x;
    int y;
};

__global__ void __launch_bounds__(kThreads) predictive_match_kernel(MatchGeometry g, TemporalWindow w,
                                                                    RasterGrid grid, DeviceMatch* out, int* counts) {
    __shared__ MatchKey keys[kSortSize];
    __shared__ MatchKey global_keys[128];
    __shared__ Center seeds[kMaxGroup];
    __shared__ Center centers[kMaxGroup];
    __shared__ int global_count;
    __shared__ int center_count;

    const int ref = blockIdx.x;
    const int cx = grid.x(ref);
    const int cy = grid.y(ref);
    const int t0 = w.t0;
    DeviceMatch* dst = out + static_cast<long long>(ref) * kMaxGroup;
    const int wanted = min(g.group, kMaxGroup);
    if (wanted <= 1) {
        write_result(dst, counts + ref, cx, cy, t0, keys, 0);
        return;
    }
    const float* anchor_plane = w.frames[t0];
    BlockTopK<kSortSize> topk;
    topk.init(keys, wanted - 1);
    scan_spatial(topk, anchor_plane, anchor_plane, g, cx, cy, t0);
    const int spatial = topk.filled;
    if (w.radius == 0 || w.ntemp == 1) {
        write_result(dst, counts + ref, cx, cy, t0, keys, spatial);
        return;
    }

    // Seeds: the reference itself followed by the best spatial matches.
    const int seed_count = min(min(1 + spatial, w.ps_num), kMaxGroup);
    for (int i = threadIdx.x; i < 128; i += blockDim.x) {
        global_keys[i] = i < spatial ? keys[i] : detail::sentinel_key();
    }
    for (int i = threadIdx.x; i < seed_count; i += blockDim.x) {
        seeds[i] = i == 0 ? Center{cx, cy} : Center{detail::key_x(keys[i - 1]), detail::key_y(keys[i - 1])};
    }
    if (threadIdx.x == 0) global_count = spatial;
    __syncthreads();

    const int first = max(0, w.valid_begin);
    const int last = w.valid_end < 0 ? w.ntemp : min(w.ntemp, w.valid_end);
    const int local_k = min(w.ps_num, g.group);
    const int pr = w.ps_range;
    const int max_x = g.width - g.block;
    const int max_y = g.height - g.block;
    const float* anchor = anchor_plane + cy * g.pitch + cx;

    for (int sign = -1; sign <= 1; sign += 2) {
        for (int i = threadIdx.x; i < seed_count; i += blockDim.x) centers[i] = seeds[i];
        if (threadIdx.x == 0) center_count = seed_count;
        __syncthreads();
        for (int dt = 1; dt <= w.radius; ++dt) {
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
            const float* plane = w.frames[t];
            topk.init(keys, local_k);
            if (ymax >= ymin && xmax >= xmin) {
                const int bw = xmax - xmin + 1;
                const int total = bw * (ymax - ymin + 1);
                for (int base = 0; base < total; base += BlockTopK<kSortSize>::kChunk) {
                    for (int i = threadIdx.x; i < BlockTopK<kSortSize>::kChunk; i += blockDim.x) {
                        const int idx = base + i;
                        MatchKey key = detail::sentinel_key();
                        if (idx < total) {
                            const int x = xmin + idx % bw;
                            const int y = ymin + idx / bw;
                            bool inside = false;
                            for (int c = 0; c < nc && !inside; ++c) {
                                inside = abs(y - centers[c].y) <= pr && abs(x - centers[c].x) <= pr;
                            }
                            if (inside) {
                                key = detail::make_key(block_ssd(anchor, plane + y * g.pitch + x, g.pitch, g.block), t,
                                                       y, x);
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

}  // namespace

RasterGrid make_raster_grid(int width, int height, int block, int step) {
    RasterGrid grid{0, 0, step, width - block, height - block};
    if (width < block || height < block || step < 1) return grid;
    grid.nx = (width - block + step - 1) / step + 1;
    grid.ny = (height - block + step - 1) / step + 1;
    return grid;
}

void spatial_match(const float* plane, const MatchGeometry& geometry, const RasterGrid& grid, DeviceMatch* out,
                   int* counts, cudaStream_t stream) {
    if (grid.count() == 0) return;
    spatial_match_kernel<<<grid.count(), kThreads, 0, stream>>>(plane, geometry, grid, out, counts);
    NSS_CUDA_CHECK_LAUNCH();
}

void predictive_match(const MatchGeometry& geometry, const TemporalWindow& window, const RasterGrid& grid,
                      DeviceMatch* out, int* counts, cudaStream_t stream) {
    if (grid.count() == 0) return;
    predictive_match_kernel<<<grid.count(), kThreads, 0, stream>>>(geometry, window, grid, out, counts);
    NSS_CUDA_CHECK_LAUNCH();
}

}  // namespace nss_cuda
