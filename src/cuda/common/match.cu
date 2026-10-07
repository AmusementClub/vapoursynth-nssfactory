// SPDX-License-Identifier: GPL-2.0-only
// Block matching kernels (see match.hpp for the contract). The block and warp
// kernels sum SSDs row-major in the same order, so their staged (shared-memory
// window) and global variants return identical matches; the lane kernels sum
// per lane and then across lanes.
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

// warp_select for a raster-ordered spatial window. There the candidate index
// order is the (y, x) order and t is common, so (sortable distance, index)
// packed in 64 bits is the same total order as the full key: one word to scan
// and to reduce, and the position is rebuilt for the winner only.
__device__ int warp_select_raster(unsigned* sortable, int total, int lane, int k, int t, const SpatialLayout& layout,
                                  MatchKey* out) {
    int filled = 0;
    for (; filled < k; ++filled) {
        unsigned long long best = ~0ull;
        for (int idx = lane; idx < total; idx += 32) {
            const unsigned long long key =
                (static_cast<unsigned long long>(sortable[idx]) << 32) | static_cast<unsigned>(idx);
            best = key < best ? key : best;
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            const unsigned long long other = __shfl_xor_sync(0xffffffffu, best, offset);
            best = other < best ? other : best;
        }
        const unsigned s = static_cast<unsigned>(best >> 32);
        if (s == kConsumed) break;  // nothing left
        const int index = static_cast<int>(best & 0xffffffffu);
        if (lane == 0) {
            out[filled] = key_of(s, t, layout.top + index / layout.ww, layout.left + index % layout.ww);
            sortable[index] = kConsumed;
        }
        __syncwarp();
    }
    return filled;
}

// Sortable distances of one lane's candidates with the B x B reference block
// held in registers (one channel).
template <int B>
__device__ __forceinline__ void warp_distances(const float* anchor, const float* window, int sw, const Window& w,
                                               const SpatialLayout& layout, SpatialLayout::Cursor cursor, int lane,
                                               int total, int cx, int cy, unsigned* sortable) {
    float ref[B * B];
#pragma unroll
    for (int r = 0; r < B; ++r) {
#pragma unroll
        for (int c = 0; c < B; ++c) ref[r * B + c] = anchor[r * sw + c];
    }
    for (int idx = lane; idx < total; idx += 32, layout.advance(cursor)) {
        const Center pos = layout.at(cursor);
        const float* b = window + (pos.y - w.top) * sw + (pos.x - w.left);
        float sum = 0.f;
#pragma unroll
        for (int r = 0; r < B; ++r) {
#pragma unroll
            for (int c = 0; c < B; ++c) {
                const float d = ref[r * B + c] - b[r * sw + c];
                sum = fmaf(d, d, sum);
            }
        }
        sortable[idx] = (pos.x == cx && pos.y == cy) ? kConsumed : detail::sortable_distance(sum);
    }
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
    // The kernel is bound by shared-memory loads, two per term. Holding the
    // reference block in registers leaves one; the terms and their order are
    // those of block_ssd_fixed.
    if (g.channels == 1 && g.block == 8) {
        warp_distances<8>(anchor, window, sw, w, layout, cursor, lane, total, cx, cy, sortable);
        __syncwarp();
        return total;
    }
    if (g.channels == 1 && g.block == 4) {
        warp_distances<4>(anchor, window, sw, w, layout, cursor, lane, total, cx, cy, sortable);
        __syncwarp();
        return total;
    }
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
    const int filled = warp_select_raster(sortable, total, lane, wanted, 0, layout, selected);
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
    int global_count = warp_select_raster(sortable, spatial_total, lane, wanted, t0, spatial, merge);

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

// Lane-group variants for C channels (one, or three matched jointly) and at
// most L matches. L lanes serve
// one reference, 32 / L references share a warp, and everything stays in
// registers. The lanes hold the reference's best L candidates in ascending
// key order (lane 0 the best); a candidate enters by shifting the entries
// behind it one lane up.
// - L <= B: lane j owns the columns j, j + L, ... of the block and a shuffle
//   tree over the L lanes totals a candidate.
// - L > B: the lanes form parts of W lanes, W the block width rounded up to
//   a power of two. Each part takes a column of its own (lane j the block
//   column j % W), so L / W candidates are summed at once and then entered
//   one after the other.
// A lane whose column lies beyond the block (widths that are not a power of
// two, as 12) adds nothing. With three channels a lane holds its columns of
// each and adds their terms into one sum, channel after channel.
// A window is scanned column by column, so that the next candidate is the
// block one row down: each lane keeps its B rows in registers and reads one
// new row per candidate. The loops are the same for every reference (the
// unclamped window; positions outside the plane never enter), so the warp
// never diverges.
//
// Keys are distance bits << 32 | y << 16 | x. A sum of squares is +0, a
// positive float or not finite, so its bit pattern, capped at +inf's, sorts
// as the CPU's distances do (the non-finite ones equal, after the others).
// Warps per block. One-warp blocks stop at half the device's warps; four
// measured 1.25x one for the spatial kernel and 1.13x for the predictive one.
constexpr int kLaneWarps = 4;
constexpr unsigned kAllLanes = 0xffffffffu;
constexpr int kLaneMaxCoordinate = 0xffff;
constexpr unsigned kLaneInfinity = 0x7f800000u;
constexpr unsigned long long kLaneEmpty = ~0ull;

template <int L>
__device__ __forceinline__ int lanes_set(unsigned ballot, int lane) {
    const unsigned mask = L == 32 ? ~0u : (1u << L) - 1u;
    return __popc((ballot >> (lane - lane % L)) & mask);
}

__device__ __forceinline__ unsigned lane_position(int x, int y) {
    return static_cast<unsigned>(y) << 16 | static_cast<unsigned>(x);
}

__device__ __forceinline__ DeviceMatch lane_match(unsigned long long key, int t) {
    const unsigned at = static_cast<unsigned>(key), bits = static_cast<unsigned>(key >> 32);
    return DeviceMatch{static_cast<int>(at & 0xffffu), static_cast<int>(at >> 16), t, __uint_as_float(bits)};
}

// Lanes that total one candidate, block columns a lane owns, and candidates
// summed at once.
template <int B, int L>
constexpr int kLaneWidth = L <= B ? L : detail::pow2_at_least(B);
template <int B, int L>
constexpr int kLaneCols = L <= B ? (B + L - 1) / L : 1;
template <int B, int L>
constexpr int kLaneParts = L / kLaneWidth<B, L>;
// Column c of the lane at `sub` within its part, and whether the block has it.
template <int B, int L>
__device__ __forceinline__ int lane_column(int sub, int c) {
    return sub % kLaneWidth<B, L> + c * kLaneWidth<B, L>;
}
template <int B, int L>
constexpr bool kLaneEven = B % kLaneWidth<B, L> == 0;

// The lane's terms of the reference block, channel by channel and row-major:
// ref[(ch * B + row) * kLaneCols + c] (the value is not used for a column the
// block does not have).
template <int B, int L, int C>
__device__ __forceinline__ void lane_reference(const float* plane, const MatchGeometry& g, int cx, int cy, int sub,
                                               float* ref) {
    constexpr int kCols = kLaneCols<B, L>;
#pragma unroll
    for (int ch = 0; ch < C; ++ch) {
        const float* at = plane + (C == 1 ? 0 : ch * g.channel_step) + static_cast<long long>(cy) * g.pitch + cx;
#pragma unroll
        for (int row = 0; row < B; ++row) {
#pragma unroll
            for (int c = 0; c < kCols; ++c) {
                ref[(ch * B + row) * kCols + c] = at[row * g.pitch + min(lane_column<B, L>(sub, c), B - 1)];
            }
        }
    }
}

// Candidates of the window [mx - lo, mx + hi] x [my - lo, my + hi] of
// `plane` into the lane-held list `held`. `skip` is a position that never
// enters (the reference itself), `enabled` switches the whole window off.
// With Dedupe a key already held is dropped: a position scanned twice (from
// two overlapping windows) has the same key both times, and when its first
// was not kept the second is not either.
template <int B, int L, int C, bool Dedupe>
__device__ __forceinline__ void lane_scan(const float* plane, const MatchGeometry& g, int mx, int my, int lo, int hi,
                                          bool enabled, unsigned skip, int lane, const float* ref,
                                          unsigned long long& held) {
    const int sub = lane % L;
    constexpr int kCols = kLaneCols<B, L>, kParts = kLaneParts<B, L>, kWidth = kLaneWidth<B, L>;
    // Columns of this lane inside the block, clamped for the reads; `has`
    // says which exist (all of them when the width divides the block).
    int column[kCols];
    bool has[kCols];
#pragma unroll
    for (int c = 0; c < kCols; ++c) {
        has[c] = kLaneEven<B, L> || lane_column<B, L>(sub, c) < B;
        column[c] = min(lane_column<B, L>(sub, c), B - 1);
    }
    const int max_x = g.width - B, max_y = g.height - B, last_row = g.height - 1;
    const int bottom = my + hi;
    for (int x0 = mx - lo; x0 <= mx + hi; x0 += kParts) {
        const int x = x0 + sub / kWidth;  // this part's column
        const bool col_ok = enabled && x >= 0 && x <= max_x && x <= mx + hi;
        const float* col = plane + min(max(x, 0), max_x);
        // v[(ch * kCols + c) * B + slot]: the rows of the current block, as a
        // ring. Before phase i, row y + k is in slot (i + 1 + k) % B for
        // k < B - 1.
        float v[C * kCols * B];
#pragma unroll
        for (int k = 0; k < B - 1; ++k) {
            const float* row = col + static_cast<long long>(min(max(my - lo + k, 0), last_row)) * g.pitch;
#pragma unroll
            for (int ch = 0; ch < C; ++ch) {
                const float* plane_row = row + (C == 1 ? 0 : ch * g.channel_step);
#pragma unroll
                for (int c = 0; c < kCols; ++c) v[(ch * kCols + c) * B + k + 1] = plane_row[column[c]];
            }
        }
        int y = my - lo;
        while (y <= bottom) {
#pragma unroll
            for (int i = 0; i < B; ++i) {
                if (y > bottom) break;
                const float* row = col + static_cast<long long>(min(max(y + B - 1, 0), last_row)) * g.pitch;
                float sum = 0.f;
#pragma unroll
                for (int ch = 0; ch < C; ++ch) {
                    const float* plane_row = row + (C == 1 ? 0 : ch * g.channel_step);
#pragma unroll
                    for (int c = 0; c < kCols; ++c) v[(ch * kCols + c) * B + i] = plane_row[column[c]];
#pragma unroll
                    for (int k = 0; k < B; ++k) {
#pragma unroll
                        for (int c = 0; c < kCols; ++c) {
                            const float d =
                                has[c] ? ref[(ch * B + k) * kCols + c] - v[(ch * kCols + c) * B + (i + 1 + k) % B] : 0.f;
                            sum = fmaf(d, d, sum);
                        }
                    }
                }
                // The same value on every lane of the candidate: the tree
                // adds the same pairs on each of them.
#pragma unroll
                for (int offset = 1; offset < kWidth; offset <<= 1) {
                    sum += __shfl_xor_sync(kAllLanes, sum, offset);
                }
                const unsigned at = lane_position(x, y);
                const bool ok = col_ok && y >= 0 && y <= max_y && at != skip;
                const unsigned long long mine =
                    ok ? static_cast<unsigned long long>(min(__float_as_uint(sum), kLaneInfinity)) << 32 | at
                       : kLaneEmpty;
#pragma unroll
                for (int part = 0; part < kParts; ++part) {
                    const unsigned long long key = kParts == 1 ? mine : __shfl_sync(kAllLanes, mine, part * kWidth, L);
                    bool enters = key < held;
                    if constexpr (Dedupe) {
                        // Every lane takes part in the ballot: not behind `enters`.
                        const int same = lanes_set<L>(__ballot_sync(kAllLanes, held == key), lane);
                        enters = enters && same == 0;
                    }
                    const unsigned long long pre = __shfl_up_sync(kAllLanes, held, 1, L);
                    if (enters) held = (sub == 0 || !(key < pre)) ? key : pre;
                }
                ++y;
            }
        }
    }
}

template <int B, int L, int C>
__global__ void __launch_bounds__(32 * kLaneWarps) spatial_match_lane_kernel(const float* plane, MatchGeometry g, RasterGrid grid,
                                                                int ref_begin, int ref_count, DeviceMatch* out,
                                                                int* counts) {
    const int lane = threadIdx.x % 32, sub = lane % L;
    const int r = (blockIdx.x * kLaneWarps + threadIdx.x / 32) * (32 / L) + lane / L;
    const bool live = r < ref_count;
    const int rr = live ? r : ref_count - 1;  // idle lanes mirror the last reference and store nothing
    const int cx = grid.x(ref_begin + rr);
    const int cy = grid.y(ref_begin + rr);
    float ref[C * B * kLaneCols<B, L>];
    lane_reference<B, L, C>(plane, g, cx, cy, sub, ref);
    unsigned long long held = kLaneEmpty;
    lane_scan<B, L, C, false>(plane, g, cx, cy, g.lo(), g.hi(), true, lane_position(cx, cy), lane, ref, held);
    const bool mine = sub < g.group - 1 && held != kLaneEmpty;
    const int filled = lanes_set<L>(__ballot_sync(kAllLanes, mine), lane);
    if (!live) return;
    DeviceMatch* dst = out + static_cast<long long>(r) * g.group;
    if (mine) dst[1 + sub] = lane_match(held, 0);
    if (sub == 0) {
        dst[0] = DeviceMatch{cx, cy, 0, 0.f};
        counts[r] = 1 + filled;
    }
}

// Predictive search in the same layout, for ps_num <= L. The lanes hold the
// global list under (distance, t, y, x), t beside the key, and per layer the
// layer's best L candidates.
template <int B, int L, int C>
__global__ void __launch_bounds__(32 * kLaneWarps) predictive_match_lane_kernel(MatchGeometry g, TemporalWindow tw, RasterGrid grid,
                                                                   int ref_begin, int ref_count, DeviceMatch* out,
                                                                   int* counts) {
    const int lane = threadIdx.x % 32, sub = lane % L;
    const int r = (blockIdx.x * kLaneWarps + threadIdx.x / 32) * (32 / L) + lane / L;
    const bool live = r < ref_count;
    const int rr = live ? r : ref_count - 1;
    const int cx = grid.x(ref_begin + rr);
    const int cy = grid.y(ref_begin + rr);
    const int t0 = tw.t0;
    const int wanted = g.group - 1;
    float ref[C * B * kLaneCols<B, L>];
    lane_reference<B, L, C>(tw.frames[t0], g, cx, cy, sub, ref);
    unsigned long long held = kLaneEmpty;
    lane_scan<B, L, C, false>(tw.frames[t0], g, cx, cy, g.lo(), g.hi(), true, lane_position(cx, cy), lane, ref, held);
    if (sub >= wanted) held = kLaneEmpty;
    int held_t = t0;

    if (tw.radius > 0 && tw.ntemp > 1) {
        // Seeds: the reference itself followed by the best spatial matches.
        const int spatial = lanes_set<L>(__ballot_sync(kAllLanes, held != kLaneEmpty), lane);
        const int seed_count = min(1 + spatial, tw.ps_num);
        const unsigned shifted = __shfl_up_sync(kAllLanes, static_cast<unsigned>(held), 1, L);
        const unsigned seeds = sub == 0 ? lane_position(cx, cy) : shifted;
        const int first = max(0, tw.valid_begin);
        const int last = tw.valid_end < 0 ? tw.ntemp : min(tw.ntemp, tw.valid_end);
        const int local_k = min(tw.ps_num, g.group);
        for (int sign = -1; sign <= 1; sign += 2) {
            unsigned centers = seeds;
            int nc = seed_count;
            for (int dt = 1; dt <= tw.radius; ++dt) {
                const int t = t0 + sign * dt;
                if (t < first || t >= last) break;
                const float* plane = tw.frames[t];
                unsigned long long local = kLaneEmpty;
                for (int i = 0; i < tw.ps_num; ++i) {
                    const unsigned c = __shfl_sync(kAllLanes, centers, i, L);
                    lane_scan<B, L, C, true>(plane, g, static_cast<int>(c & 0xffffu), static_cast<int>(c >> 16), tw.ps_range,
                                          tw.ps_range, i < nc, ~0u, lane, ref, local);
                }
                const int found = lanes_set<L>(__ballot_sync(kAllLanes, sub < local_k && local != kLaneEmpty), lane);
                // The layer's best into the global list, one at a time.
                for (int i = 0; i < tw.ps_num; ++i) {
                    const unsigned long long e = __shfl_sync(kAllLanes, local, i, L);
                    const unsigned long long pre = __shfl_up_sync(kAllLanes, held, 1, L);
                    const int pre_t = __shfl_up_sync(kAllLanes, held_t, 1, L);
                    const unsigned bits = static_cast<unsigned>(e >> 32), at = static_cast<unsigned>(e);
                    const auto before = [&](unsigned long long h, int ht) {
                        const unsigned h_bits = static_cast<unsigned>(h >> 32);
                        return bits < h_bits || (bits == h_bits && (t < ht || (t == ht && at < static_cast<unsigned>(h))));
                    };
                    if (i < found && before(held, held_t)) {
                        const bool head = sub == 0 || !before(pre, pre_t);
                        held = head ? e : pre;
                        held_t = head ? t : pre_t;
                    }
                }
                centers = static_cast<unsigned>(local);
                nc = found;  // none: the later layers of this direction find nothing either
            }
        }
    }
    const bool mine = sub < wanted && held != kLaneEmpty;
    const int filled = lanes_set<L>(__ballot_sync(kAllLanes, mine), lane);
    if (!live) return;
    DeviceMatch* dst = out + static_cast<long long>(r) * g.group;
    if (mine) dst[1 + sub] = lane_match(held, held_t);
    if (sub == 0) {
        dst[0] = DeviceMatch{cx, cy, t0, 0.f};
        counts[r] = 1 + filled;
    }
}

// Lanes per reference for the shapes the lane kernels serve (0: none): the
// fewest of 8, 16 and 32 that hold the matches and the layer's best. Block
// 32 takes all 32 lanes (a lane then holds one column: with fewer lanes the
// rows of several columns do not fit registers). Three joint channels are
// served up to block 8, where a lane's rows of all three still do.
int lane_group(const MatchGeometry& g, int ps_num) {
    if ((g.channels != 1 && g.channels != 3) || g.group < 2) return 0;
    switch (g.block) {
    case 1: case 2: case 4: case 8: break;
    case 12: case 16: case 32:
        if (g.channels != 1) return 0;
        break;
    default: return 0;
    }
    if (g.width > kLaneMaxCoordinate || g.height > kLaneMaxCoordinate) return 0;
    const int need = std::max(g.group - 1, ps_num);
    if (need > 32) return 0;
    return g.block == 32 ? 32 : need <= 8 ? 8 : need <= 16 ? 16 : 32;
}

template <int B, int L, int C>
void launch_lane(const float* plane, const MatchGeometry& g, const TemporalWindow* tw, const RasterGrid& grid,
                 int ref_begin, int ref_count, DeviceMatch* out, int* counts, cudaStream_t stream) {
    const int per_block = kLaneWarps * (32 / L);
    const int blocks = (ref_count + per_block - 1) / per_block;
    if (tw) {
        predictive_match_lane_kernel<B, L, C><<<blocks, 32 * kLaneWarps, 0, stream>>>(g, *tw, grid, ref_begin, ref_count,
                                                                                   out, counts);
    } else {
        spatial_match_lane_kernel<B, L, C><<<blocks, 32 * kLaneWarps, 0, stream>>>(plane, g, grid, ref_begin,
                                                                                   ref_count, out, counts);
    }
}

bool launch_lanes(const float* plane, const MatchGeometry& g, const TemporalWindow* tw, const RasterGrid& grid,
                  int ref_begin, int ref_count, DeviceMatch* out, int* counts, cudaStream_t stream) {
    switch ((g.channels == 3 ? 10000 : 0) + g.block * 100 + lane_group(g, tw ? tw->ps_num : 0)) {
#define NSS_LANE(B, L) \
    case B * 100 + L: launch_lane<B, L, 1>(plane, g, tw, grid, ref_begin, ref_count, out, counts, stream); break;
#define NSS_LANE_JOINT(B, L) \
    case 10000 + B * 100 + L: launch_lane<B, L, 3>(plane, g, tw, grid, ref_begin, ref_count, out, counts, stream); break;
    NSS_LANE(1, 8) NSS_LANE(1, 16) NSS_LANE(1, 32)
    NSS_LANE(2, 8) NSS_LANE(2, 16) NSS_LANE(2, 32)
    NSS_LANE(4, 8) NSS_LANE(4, 16) NSS_LANE(4, 32)
    NSS_LANE(8, 8) NSS_LANE(8, 16) NSS_LANE(8, 32)
    NSS_LANE(12, 8) NSS_LANE(12, 16) NSS_LANE(12, 32)
    NSS_LANE(16, 8) NSS_LANE(16, 16) NSS_LANE(16, 32)
    NSS_LANE(32, 32)
    NSS_LANE_JOINT(1, 8) NSS_LANE_JOINT(1, 16) NSS_LANE_JOINT(1, 32)
    NSS_LANE_JOINT(2, 8) NSS_LANE_JOINT(2, 16) NSS_LANE_JOINT(2, 32)
    NSS_LANE_JOINT(4, 8) NSS_LANE_JOINT(4, 16) NSS_LANE_JOINT(4, 32)
    NSS_LANE_JOINT(8, 8) NSS_LANE_JOINT(8, 16) NSS_LANE_JOINT(8, 32)
#undef NSS_LANE
#undef NSS_LANE_JOINT
    default: return false;
    }
    NSS_CUDA_CHECK_LAUNCH();
    return true;
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
    if (launch_lanes(plane, geometry, nullptr, grid, ref_begin, ref_count, out, counts, stream)) return;
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
    if (launch_lanes(nullptr, geometry, &window, grid, ref_begin, ref_count, out, counts, stream)) return;
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
