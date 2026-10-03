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

// Row-major sum of squared differences; every path uses this exact order.
// Common block sizes get fully unrolled loops.
template <int B>
__device__ __forceinline__ float block_ssd_fixed(const float* a, int a_pitch, const float* b, int b_pitch) {
    float sum = 0.f;
#pragma unroll
    for (int r = 0; r < B; ++r) {
#pragma unroll
        for (int c = 0; c < B; ++c) {
            const float d = a[r * a_pitch + c] - b[r * b_pitch + c];
            sum = fmaf(d, d, sum);
        }
    }
    return sum;
}

__device__ __forceinline__ float block_ssd(const float* a, int a_pitch, const float* b, int b_pitch, int block) {
    switch (block) {
    case 4: return block_ssd_fixed<4>(a, a_pitch, b, b_pitch);
    case 8: return block_ssd_fixed<8>(a, a_pitch, b, b_pitch);
    case 16: return block_ssd_fixed<16>(a, a_pitch, b, b_pitch);
    default: break;
    }
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

// Sum over channels of the per-channel SSDs (nss::ssd_nch); one channel is
// exactly block_ssd.
__device__ __forceinline__ float group_ssd(const float* a, int a_pitch, long long a_step, const float* b, int b_pitch,
                                           long long b_step, const MatchGeometry& g) {
    float sum = block_ssd(a, a_pitch, b, b_pitch, g.block);
    for (int c = 1; c < g.channels; ++c) sum += block_ssd(a + c * a_step, a_pitch, b + c * b_step, b_pitch, g.block);
    return sum;
}

struct Window {
    int left, top, right, bottom;  // candidate position range (inclusive)
};

__device__ __forceinline__ Window spatial_window(const MatchGeometry& g, int cx, int cy) {
    return Window{max(cx - g.lo(), 0), max(cy - g.lo(), 0), min(cx + g.hi(), g.width - g.block),
                  min(cy + g.hi(), g.height - g.block)};
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
                             int base_pitch, long long step, int base_x, int base_y, const MatchGeometry& g,
                             const Window& w, int cx, int cy, int t) {
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
                    key = detail::make_key(group_ssd(anchor, anchor_pitch, step, cand, base_pitch, step, g), t, y, x);
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
            const long long at = static_cast<long long>(w.top + i / sw) * g.pitch + w.left + i % sw;
            for (int c = 0; c < g.channels; ++c) window[c * sw * sh + i] = plane[c * g.channel_step + at];
        }
        __syncthreads();
        const float* anchor = window + (cy - w.top) * sw + (cx - w.left);
        scan_spatial(topk, anchor, sw, window, sw, sw * sh, w.left, w.top, g, w, cx, cy, 0);
    } else {
        const float* anchor = plane + static_cast<long long>(cy) * g.pitch + cx;
        scan_spatial(topk, anchor, g.pitch, plane, g.pitch, g.channel_step, 0, 0, g, w, cx, cy, 0);
    }
    write_result(dst, counts + ref, cx, cy, 0, keys, topk.filled);
}

// Warp-per-reference variants for small top-K (the common BM3D/WNNM case).
// Each warp stages its search window in its own shared memory and keeps one
// 4-byte sortable distance per candidate; positions are recomputed from the
// candidate index when keys are compared, which keeps the block small enough
// for good occupancy. The K smallest keys are extracted by K warp-wide argmin
// rounds. Keys are a total order, so the result equals the block top-K path.
// A block holds as many warps (references) as fit the shared-memory budget:
// kWarpRefs for the BM3D/WNNM windows, fewer for NLH's larger ones.
constexpr int kWarpRefs = kThreads / 32;
constexpr int kWarpMaxK = 16;
constexpr unsigned kConsumed = 0xffffffffu;  // sortable_distance never returns this

struct Center {
    int x;
    int y;
};

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

__device__ __forceinline__ MatchKey key_of(unsigned sortable, int t, int y, int x) {
    return MatchKey{(static_cast<unsigned long long>(sortable) << 32) | static_cast<unsigned>(t),
                    (static_cast<unsigned long long>(static_cast<unsigned>(y)) << 32) | static_cast<unsigned>(x)};
}

// Candidate index -> position, walked in steps of 32 (one lane's candidates)
// without per-candidate divisions. Spatial: raster order of the clamped window.
struct SpatialLayout {
    int left, top, ww;
    struct Cursor {
        int col, row;
    };
    __device__ __forceinline__ Cursor begin(int idx) const { return Cursor{idx % ww, idx / ww}; }
    __device__ __forceinline__ void advance(Cursor& c) const {
        c.col += 32;
        while (c.col >= ww) {
            c.col -= ww;
            ++c.row;
        }
    }
    __device__ __forceinline__ Center at(const Cursor& c) const { return Center{left + c.col, top + c.row}; }
};
// Temporal layer: side x side windows around each center, center-major.
struct LayerLayout {
    const Center* centers;
    int pr, side;
    struct Cursor {
        int center, col, row;
    };
    __device__ __forceinline__ Cursor begin(int idx) const {
        const int o = idx % (side * side);
        return Cursor{idx / (side * side), o % side, o / side};
    }
    __device__ __forceinline__ void advance(Cursor& c) const {
        c.col += 32;
        while (c.col >= side) {
            c.col -= side;
            if (++c.row == side) {
                c.row = 0;
                ++c.center;
            }
        }
    }
    __device__ __forceinline__ Center at(const Cursor& c) const {
        const Center m = centers[c.center];
        return Center{m.x - pr + c.col, m.y - pr + c.row};
    }
};

// K smallest of the candidates into out[0, k) in ascending key order
// (consumes the selected entries). Warp-uniform return value.
template <class Layout>
__device__ int warp_select(unsigned* sortable, int total, int lane, int k, int t, const Layout& layout,
                           MatchKey* out) {
    int filled = 0;
    for (; filled < k; ++filled) {
        MatchKey best = detail::sentinel_key();
        int index = -1;
        auto cursor = layout.begin(lane);
        for (int idx = lane; idx < total; idx += 32, layout.advance(cursor)) {
            const unsigned s = sortable[idx];
            if (s == kConsumed) continue;
            const Center pos = layout.at(cursor);
            const MatchKey key = key_of(s, t, pos.y, pos.x);
            if (detail::key_less(key, best)) {
                best = key;
                index = idx;
            }
        }
        warp_min(best, index);
        if (index < 0) break;
        if (lane == 0) {
            out[filled] = best;
            sortable[index] = kConsumed;
        }
        __syncwarp();
    }
    return filled;
}

// Stages the spatial window and fills the sortable distances of the spatial
// candidates of (cx, cy); returns the layout and candidate count.
__device__ __forceinline__ int warp_spatial(const float* plane, const MatchGeometry& g, int cx, int cy, int lane,
                                            float* window, unsigned* sortable, SpatialLayout& layout,
                                            const float*& anchor, int& anchor_pitch, int& anchor_step) {
    const Window w = spatial_window(g, cx, cy);
    const int sw = w.right - w.left + g.block;
    const int sh = w.bottom - w.top + g.block;
    for (int i = lane; i < sw * sh; i += 32) {
        const long long at = static_cast<long long>(w.top + i / sw) * g.pitch + w.left + i % sw;
        for (int c = 0; c < g.channels; ++c) window[c * sw * sh + i] = plane[c * g.channel_step + at];
    }
    __syncwarp();
    layout = SpatialLayout{w.left, w.top, w.right - w.left + 1};
    anchor = window + (cy - w.top) * sw + (cx - w.left);
    anchor_pitch = sw;
    anchor_step = sw * sh;
    const int total = layout.ww * (w.bottom - w.top + 1);
    auto cursor = layout.begin(lane);
    for (int idx = lane; idx < total; idx += 32, layout.advance(cursor)) {
        const Center pos = layout.at(cursor);
        sortable[idx] = (pos.x == cx && pos.y == cy)
                            ? kConsumed
                            : detail::sortable_distance(group_ssd(
                                  anchor, sw, sw * sh, window + (pos.y - w.top) * sw + (pos.x - w.left), sw, sw * sh, g));
    }
    __syncwarp();
    return total;
}

__global__ void __launch_bounds__(kThreads) spatial_match_warp_kernel(const float* plane, MatchGeometry g,
                                                                      RasterGrid grid, int ref_begin, int ref_count,
                                                                      DeviceMatch* out, int* counts, int window_floats,
                                                                      int capacity) {
    extern __shared__ unsigned char shared[];
    const int warps = blockDim.x / 32;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int ref = blockIdx.x * warps + warp;
    if (ref >= ref_count) return;  // warp-uniform; only warp barriers below
    auto* selected = reinterpret_cast<MatchKey*>(shared) + warp * kWarpMaxK;
    auto* sortable = reinterpret_cast<unsigned*>(reinterpret_cast<MatchKey*>(shared) + warps * kWarpMaxK) +
                     warp * capacity;
    float* window = reinterpret_cast<float*>(reinterpret_cast<unsigned*>(
                        reinterpret_cast<MatchKey*>(shared) + warps * kWarpMaxK) + warps * capacity) +
                    warp * window_floats;
    const int cx = grid.x(ref_begin + ref);
    const int cy = grid.y(ref_begin + ref);
    DeviceMatch* dst = out + static_cast<long long>(ref) * g.group;
    const int wanted = min(g.group, kMaxGroup) - 1;
    SpatialLayout layout{};
    const float* anchor = nullptr;
    int anchor_pitch = 0, anchor_step = 0;
    const int total = warp_spatial(plane, g, cx, cy, lane, window, sortable, layout, anchor, anchor_pitch, anchor_step);
    const int filled = warp_select(sortable, total, lane, wanted, 0, layout, selected);
    for (int i = lane; i < filled; i += 32) {
        const MatchKey& k = selected[i];
        dst[1 + i] = DeviceMatch{detail::key_x(k), detail::key_y(k), detail::key_t(k), detail::key_distance(k)};
    }
    if (lane == 0) {
        dst[0] = DeviceMatch{cx, cy, 0, 0.f};
        counts[ref] = 1 + filled;
    }
}

// Predictive search for wanted - 1 <= 16 and ps_num <= 16. Per warp: a
// 32-entry merge array (global top in [0, 16), the current layer's local top
// in [16, 32)), a selection scratch, centers and seeds, the sortable
// distances and the staged spatial window (which also holds the anchor).
__global__ void __launch_bounds__(kThreads) predictive_match_warp_kernel(MatchGeometry g, TemporalWindow tw,
                                                                         RasterGrid grid, int ref_begin,
                                                                         int ref_count, DeviceMatch* out, int* counts,
                                                                         int window_floats, int capacity) {
    extern __shared__ unsigned char shared[];
    constexpr int kSlots = kWarpMaxK;
    const int warps = blockDim.x / 32;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int ref = blockIdx.x * warps + warp;
    if (ref >= ref_count) return;  // warp-uniform; only warp barriers below
    auto* key_base = reinterpret_cast<MatchKey*>(shared);
    MatchKey* merge = key_base + warp * 3 * kSlots;
    MatchKey* scratch = merge + 2 * kSlots;
    auto* center_base = reinterpret_cast<Center*>(key_base + warps * 3 * kSlots);
    Center* centers = center_base + warp * 2 * kSlots;
    Center* seeds = centers + kSlots;
    auto* sortable_base = reinterpret_cast<unsigned*>(center_base + warps * 2 * kSlots);
    unsigned* sortable = sortable_base + warp * capacity;
    float* window = reinterpret_cast<float*>(sortable_base + warps * capacity) + warp * window_floats;

    const int cx = grid.x(ref_begin + ref);
    const int cy = grid.y(ref_begin + ref);
    const int t0 = tw.t0;
    DeviceMatch* dst = out + static_cast<long long>(ref) * g.group;
    const int wanted = min(g.group, kMaxGroup) - 1;
    SpatialLayout spatial{};
    const float* anchor = nullptr;
    int anchor_pitch = 0, anchor_step = 0;
    const int spatial_total =
        warp_spatial(tw.frames[t0], g, cx, cy, lane, window, sortable, spatial, anchor, anchor_pitch, anchor_step);
    merge[lane] = detail::sentinel_key();  // 32 lanes cover the 32 merge entries
    __syncwarp();
    int global_count = warp_select(sortable, spatial_total, lane, wanted, t0, spatial, merge);

    if (tw.radius > 0 && tw.ntemp > 1) {
        const int seed_count = min(min(1 + global_count, tw.ps_num), kSlots);
        if (lane < seed_count) {
            seeds[lane] = lane == 0 ? Center{cx, cy} : Center{detail::key_x(merge[lane - 1]), detail::key_y(merge[lane - 1])};
        }
        __syncwarp();
        const int first = max(0, tw.valid_begin);
        const int last = tw.valid_end < 0 ? tw.ntemp : min(tw.ntemp, tw.valid_end);
        const int local_k = min(tw.ps_num, g.group);
        const int pr = tw.ps_range;
        const int side = 2 * pr + 1;
        const int max_x = g.width - g.block;
        const int max_y = g.height - g.block;
        const LayerLayout layer{centers, pr, side};
        for (int sign = -1; sign <= 1; sign += 2) {
            if (lane < seed_count) centers[lane] = seeds[lane];
            __syncwarp();
            int nc = seed_count;
            for (int dt = 1; dt <= tw.radius; ++dt) {
                const int t = t0 + sign * dt;
                if (t < first || t >= last) break;
                const float* plane = tw.frames[t];
                // Union of the centers' windows: a position belongs to the
                // first center whose window contains it.
                const int total = nc * side * side;
                auto cursor = layer.begin(lane);
                for (int idx = lane; idx < total; idx += 32, layer.advance(cursor)) {
                    const int c = cursor.center;
                    const Center pos = layer.at(cursor);
                    bool use = pos.x >= 0 && pos.x <= max_x && pos.y >= 0 && pos.y <= max_y;
                    for (int e = 0; e < c && use; ++e) {
                        use = abs(pos.x - centers[e].x) > pr || abs(pos.y - centers[e].y) > pr;
                    }
                    sortable[idx] = use ? detail::sortable_distance(group_ssd(
                                              anchor, anchor_pitch, anchor_step,
                                              plane + static_cast<long long>(pos.y) * g.pitch + pos.x, g.pitch,
                                              g.channel_step, g))
                                        : kConsumed;
                }
                __syncwarp();
                const int local = warp_select(sortable, total, lane, local_k, t, layer, merge + kSlots);
                if (lane >= kSlots + local) merge[lane] = detail::sentinel_key();
                __syncwarp();
                if (lane < local) {
                    centers[lane] = Center{detail::key_x(merge[kSlots + lane]), detail::key_y(merge[kSlots + lane])};
                }
                nc = local;
                // Global merge over the 32 lane-held keys.
                MatchKey mine = merge[lane];
                int merged = 0;
                for (; merged < wanted; ++merged) {
                    MatchKey best = mine;
                    int index = lane;
                    warp_min(best, index);
                    if (best.hi == ~0ull && best.lo == ~0ull) break;
                    if (lane == 0) scratch[merged] = best;
                    if (lane == index) mine = detail::sentinel_key();
                }
                __syncwarp();
                if (lane < kSlots) merge[lane] = lane < merged ? scratch[lane] : detail::sentinel_key();
                __syncwarp();
                global_count = merged;
                if (local == 0) break;
            }
        }
    }
    for (int i = lane; i < global_count; i += 32) {
        const MatchKey& k = merge[i];
        dst[1 + i] = DeviceMatch{detail::key_x(k), detail::key_y(k), detail::key_t(k), detail::key_distance(k)};
    }
    if (lane == 0) {
        dst[0] = DeviceMatch{cx, cy, t0, 0.f};
        counts[ref] = 1 + global_count;
    }
}

// K bounds the group size and ps_num (64 for the BM3D family, 256 for TWSC).
template <int K>
__global__ void __launch_bounds__(kThreads) predictive_match_kernel(MatchGeometry g, TemporalWindow tw,
                                                                    RasterGrid grid, int ref_begin, DeviceMatch* out,
                                                                    int* counts) {
    constexpr int N = kTemporalSort;
    __shared__ MatchKey keys[N];
    __shared__ MatchKey global_keys[2 * K];
    __shared__ Center seeds[K];
    __shared__ Center centers[K];
    __shared__ int global_count;
    __shared__ int center_count;

    const int ref = blockIdx.x;
    const int cx = grid.x(ref_begin + ref);
    const int cy = grid.y(ref_begin + ref);
    const int t0 = tw.t0;
    DeviceMatch* dst = out + static_cast<long long>(ref) * g.group;
    const int wanted = min(g.group, K);
    if (wanted <= 1) {
        write_result(dst, counts + ref, cx, cy, t0, keys, 0);
        return;
    }
    const float* anchor_plane = tw.frames[t0];
    const float* anchor = anchor_plane + static_cast<long long>(cy) * g.pitch + cx;
    BlockTopK<N> topk;
    topk.init(keys, wanted - 1);
    scan_spatial(topk, anchor, g.pitch, anchor_plane, g.pitch, g.channel_step, 0, 0, g, spatial_window(g, cx, cy), cx,
                 cy, t0);
    const int spatial = topk.filled;
    if (tw.radius == 0 || tw.ntemp == 1) {
        write_result(dst, counts + ref, cx, cy, t0, keys, spatial);
        return;
    }

    // Seeds: the reference itself followed by the best spatial matches.
    const int seed_count = min(min(1 + spatial, tw.ps_num), K);
    for (int i = threadIdx.x; i < 2 * K; i += blockDim.x) {
        global_keys[i] = i < spatial ? keys[i] : detail::sentinel_key();
    }
    for (int i = threadIdx.x; i < seed_count; i += blockDim.x) {
        seeds[i] = i == 0 ? Center{cx, cy} : Center{detail::key_x(keys[i - 1]), detail::key_y(keys[i - 1])};
    }
    if (threadIdx.x == 0) global_count = spatial;
    __syncthreads();

    const int first = max(0, tw.valid_begin);
    const int last = tw.valid_end < 0 ? tw.ntemp : min(tw.ntemp, tw.valid_end);
    const int local_k = min(min(tw.ps_num, g.group), K);
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
                                key = detail::make_key(
                                    group_ssd(anchor, g.pitch, g.channel_step, cand, g.pitch, g.channel_step, g), t, y, x);
                            }
                        }
                        topk.candidate(i) = key;
                    }
                    __syncthreads();
                    topk.merge();
                }
            }
            const int local = topk.filled;
            // Global merge: current global (< K) + this layer's local (<= K).
            const int gc = global_count;
            for (int i = threadIdx.x; i < 2 * K; i += blockDim.x) {
                if (i >= gc && i < K) global_keys[i] = detail::sentinel_key();
                if (i >= K) global_keys[i] = i - K < local ? keys[i - K] : detail::sentinel_key();
            }
            __syncthreads();
            detail::bitonic_sort<2 * K>(global_keys);
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
    const int span = g.lo() + g.hi();
    const std::size_t window = static_cast<std::size_t>(span + g.block) * (span + g.block) * g.channels * sizeof(float);
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
    const int span = geometry.lo() + geometry.hi();
    const int candidates = (span + 1) * (span + 1);
    const int wanted = std::min(geometry.group, kMaxGroup) - 1;
    const int window_floats = (span + geometry.block) * (span + geometry.block) * geometry.channels;
    const std::size_t warp_bytes =
        kWarpMaxK * sizeof(MatchKey) + candidates * sizeof(unsigned) + window_floats * sizeof(float);
    const int warps = static_cast<int>(std::min<std::size_t>(kWarpRefs, kSharedBudget / warp_bytes));
    if (wanted >= 1 && wanted <= kWarpMaxK && warps >= 1) {
        spatial_match_warp_kernel<<<(ref_count + warps - 1) / warps, 32 * warps, warps * warp_bytes, stream>>>(
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
    const int span = geometry.lo() + geometry.hi();
    const int wanted = std::min(geometry.group, kMaxGroup) - 1;
    const int side = 2 * window.ps_range + 1;
    const int candidates = std::max((span + 1) * (span + 1), std::min(window.ps_num, kWarpMaxK) * side * side);
    const int window_floats = (span + geometry.block) * (span + geometry.block) * geometry.channels;
    const std::size_t warp_bytes = 3 * kWarpMaxK * sizeof(MatchKey) + 2 * kWarpMaxK * sizeof(Center) +
                                   candidates * sizeof(unsigned) + window_floats * sizeof(float);
    const int warps = static_cast<int>(std::min<std::size_t>(kWarpRefs, kSharedBudget / warp_bytes));
    if (wanted >= 1 && wanted <= kWarpMaxK && window.ps_num <= kWarpMaxK && warps >= 1) {
        predictive_match_warp_kernel<<<(ref_count + warps - 1) / warps, 32 * warps, warps * warp_bytes, stream>>>(
            geometry, window, grid, ref_begin, ref_count, out, counts, window_floats, candidates);
        NSS_CUDA_CHECK_LAUNCH();
        return;
    }
    if (geometry.group <= 64 && window.ps_num <= 64) {
        predictive_match_kernel<64><<<ref_count, kThreads, 0, stream>>>(geometry, window, grid, ref_begin, out, counts);
    } else {
        predictive_match_kernel<256><<<ref_count, kThreads, 0, stream>>>(geometry, window, grid, ref_begin, out, counts);
    }
    NSS_CUDA_CHECK_LAUNCH();
}

}  // namespace nss_cuda
