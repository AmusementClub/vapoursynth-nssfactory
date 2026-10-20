// SPDX-License-Identifier: GPL-2.0-only
// Block matching kernels (see match.hpp for the contract). The block and warp
// kernels sum SSDs row-major in the same order, so their staged (shared-memory
// window) and global variants return identical matches; the lane kernels sum
// per lane and then across lanes.
#include "cuda/common/match.hpp"
#include "cuda/common/topk.cuh"
#include "cuda/runtime/error.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <type_traits>

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
    topk.init(keys, wanted - 1, detail::keys_pack(g.width, g.height, 1));
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

// Block per reference for large groups: one 4-byte sortable distance per
// candidate, then the K smallest of the keys (distance, candidate index) by
// a radix select, one byte a round from the top, instead of sorting them. In
// the raster window the index order is the (y, x) order and t is common, so
// this is the order of the full key and the result equals the block top-K
// path. The counts of a round are integers: their sum does not depend on the
// order of the atomic additions, nor does the result on the slot a selected
// key takes before it is ranked.
constexpr int kSelectThreads = 256;  // one histogram bin per thread
constexpr int kSelectMaxCandidates = 1 << 16;
constexpr unsigned kConsumed = 0xffffffffu;  // sortable_distance never returns this

__host__ __device__ constexpr std::size_t select_fixed_bytes(int group, int candidates) {
    return (group + (group & 1)) * sizeof(unsigned long long) + (kSelectThreads + kSelectThreads / 32 + 4 + candidates) * sizeof(unsigned);
}

template <int B>
__device__ __forceinline__ void select_distances(const float* anchor, const float* plane, int pitch, const Window& w,
                                                 int ww, int total, int cx, int cy, unsigned* sortable) {
    float ref[B * B];
#pragma unroll
    for (int r = 0; r < B; ++r) {
#pragma unroll
        for (int c = 0; c < B; ++c) ref[r * B + c] = anchor[r * B + c];
    }
    for (int idx = threadIdx.x; idx < total; idx += kSelectThreads) {
        const int x = w.left + idx % ww;
        const int y = w.top + idx / ww;
        const float* b = plane + static_cast<long long>(y) * pitch + x;
        float sum = 0.f;
#pragma unroll
        for (int r = 0; r < B; ++r) {
#pragma unroll
            for (int c = 0; c < B; ++c) {
                const float d = ref[r * B + c] - b[r * pitch + c];
                sum = fmaf(d, d, sum);
            }
        }
        sortable[idx] = (x == cx && y == cy) ? kConsumed : detail::sortable_distance(sum);
    }
}

__global__ void __launch_bounds__(kSelectThreads) spatial_match_select_kernel(const float* plane, MatchGeometry g,
                                                                              RasterGrid grid, int ref_begin,
                                                                              DeviceMatch* out, int* counts,
                                                                              int capacity) {
    extern __shared__ unsigned char shared[];
    const int lane = threadIdx.x;
    const int ref = blockIdx.x;
    const int cx = grid.x(ref_begin + ref);
    const int cy = grid.y(ref_begin + ref);
    DeviceMatch* dst = out + static_cast<long long>(ref) * g.group;
    const Window w = spatial_window(g, cx, cy);
    const int ww = w.right - w.left + 1;
    const int total = ww * (w.bottom - w.top + 1);
    const int group = min(g.group, kMaxGroup);
    const int wanted = min(group, total) - 1;  // the reference itself is not a candidate
    if (lane == 0) {
        dst[0] = DeviceMatch{cx, cy, 0, 0.f};
        counts[ref] = 1 + max(wanted, 0);
    }
    if (wanted <= 0) return;
    auto* list = reinterpret_cast<unsigned long long*>(shared);
    unsigned* histogram = reinterpret_cast<unsigned*>(list + group + (group & 1));
    unsigned* warp_total = histogram + kSelectThreads;
    unsigned* state = warp_total + kSelectThreads / 32;  // digit, keys below it, exact, slots taken
    unsigned* sortable = state + 4;
    // The reference block in shared memory; the candidates are read where
    // they are (a staged window measured the same).
    float* anchor = reinterpret_cast<float*>(sortable + capacity);
    const int area = g.block * g.block;
    for (int i = lane; i < g.channels * area; i += kSelectThreads) {
        const int c = i / area, at = i % area;
        anchor[i] = plane[c * g.channel_step + static_cast<long long>(cy + at / g.block) * g.pitch + cx + at % g.block];
    }
    histogram[lane] = 0;
    if (lane == 0) state[3] = 0;
    __syncthreads();
    if (g.channels == 1 && g.block == 8) {
        select_distances<8>(anchor, plane, g.pitch, w, ww, total, cx, cy, sortable);
    } else if (g.channels == 1 && g.block == 4) {
        select_distances<4>(anchor, plane, g.pitch, w, ww, total, cx, cy, sortable);
    } else {
        for (int idx = lane; idx < total; idx += kSelectThreads) {
            const int x = w.left + idx % ww;
            const int y = w.top + idx / ww;
            const float* cand = plane + static_cast<long long>(y) * g.pitch + x;
            sortable[idx] = (x == cx && y == cy)
                                ? kConsumed
                                : detail::sortable_distance(group_ssd(anchor, g.block, area, cand, g.pitch, g.channel_step, g));
        }
    }
    __syncthreads();

    // The key (distance << 16 | index) of rank `wanted`, a byte a round:
    // `prefix` is its leading bytes so far and `need` its rank among the keys
    // that share them. A round ends the search when the keys up to its byte
    // are exactly the wanted ones (at the last byte they are: keys differ).
    unsigned long long prefix = 0;
    int need = wanted, shift = 40;
    for (;; shift -= 8) {
        for (int idx = lane; idx < total; idx += kSelectThreads) {
            const unsigned long long key = static_cast<unsigned long long>(sortable[idx]) << 16 | static_cast<unsigned>(idx);
            if (key >> (shift + 8) == prefix) atomicAdd(histogram + static_cast<int>(key >> shift & 255), 1u);
        }
        __syncthreads();
        const unsigned count = histogram[lane];
        unsigned through = count;  // keys of the bytes up to this thread's
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const unsigned other = __shfl_up_sync(0xffffffffu, through, offset);
            if (lane % 32 >= offset) through += other;
        }
        if (lane % 32 == 31) warp_total[lane / 32] = through;
        histogram[lane] = 0;
        __syncthreads();
        for (int warp = 0; warp < lane / 32; ++warp) through += warp_total[warp];
        if (static_cast<int>(through - count) < need && need <= static_cast<int>(through)) {
            state[0] = lane;
            state[1] = through - count;
            state[2] = static_cast<int>(through) == need;
        }
        __syncthreads();
        prefix = prefix << 8 | state[0];
        need -= static_cast<int>(state[1]);
        if (state[2] || shift == 0) break;
    }
    for (int idx = lane; idx < total; idx += kSelectThreads) {
        const unsigned long long key = static_cast<unsigned long long>(sortable[idx]) << 16 | static_cast<unsigned>(idx);
        if (key >> shift <= prefix) list[atomicAdd(state + 3, 1u)] = key;
    }
    __syncthreads();
    for (int i = lane; i < wanted; i += kSelectThreads) {
        const unsigned long long key = list[i];
        int rank = 0;
        for (int k = 0; k < wanted; ++k) rank += list[k] < key;
        const int idx = static_cast<int>(key & 0xffff);
        dst[1 + rank] = DeviceMatch{w.left + idx % ww, w.top + idx / ww, 0,
                                    detail::key_distance(MatchKey{key >> 16 << 32, 0})};
    }
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
                    // Most candidates enter no list: the shift is skipped then.
                    if (__any_sync(kAllLanes, enters)) {
                        const unsigned long long pre = __shfl_up_sync(kAllLanes, held, 1, L);
                        if (enters) held = (sub == 0 || !(key < pre)) ? key : pre;
                    }
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
// served up to block 8, where a lane's rows of all three still do. Every
// block size up to 16 has an instance: the warp kernel that served the
// others took 1.87x as long (block 10, docs/cuda-tools-all-20261017.md N13).
int lane_group(const MatchGeometry& g, int ps_num) {
    if ((g.channels != 1 && g.channels != 3) || g.group < 2) return 0;
    switch (g.block) {
    case 1: case 2: case 3: case 4: case 5: case 6: case 7: case 8: break;
    case 9: case 10: case 11: case 12: case 13: case 14: case 15: case 16: case 32:
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
#define NSS_LANES(B) NSS_LANE(B, 8) NSS_LANE(B, 16) NSS_LANE(B, 32)
#define NSS_LANES_JOINT(B) NSS_LANE_JOINT(B, 8) NSS_LANE_JOINT(B, 16) NSS_LANE_JOINT(B, 32)
    NSS_LANES(1) NSS_LANES(2) NSS_LANES(3) NSS_LANES(4) NSS_LANES(5) NSS_LANES(6) NSS_LANES(7) NSS_LANES(8)
    NSS_LANES(9) NSS_LANES(10) NSS_LANES(11) NSS_LANES(12) NSS_LANES(13) NSS_LANES(14) NSS_LANES(15) NSS_LANES(16)
    NSS_LANE(32, 32)
    NSS_LANES_JOINT(1) NSS_LANES_JOINT(2) NSS_LANES_JOINT(3) NSS_LANES_JOINT(4)
    NSS_LANES_JOINT(5) NSS_LANES_JOINT(6) NSS_LANES_JOINT(7) NSS_LANES_JOINT(8)
#undef NSS_LANES
#undef NSS_LANES_JOINT
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
    const bool pack = detail::keys_pack(g.width, g.height, tw.ntemp);
    BlockTopK<N> topk;
    topk.init(keys, wanted - 1, pack);
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
            topk.init(keys, local_k, pack);
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

bool launch_select(const float* plane, const MatchGeometry& g, const RasterGrid& grid, int ref_begin, int ref_count,
                   DeviceMatch* out, int* counts, cudaStream_t stream) {
    const int span = g.lo() + g.hi();
    const int candidates = (span + 1) * (span + 1);
    if (candidates > kSelectMaxCandidates) return false;
    const std::size_t bytes = select_fixed_bytes(std::min(g.group, kMaxGroup), candidates) +
                              static_cast<std::size_t>(g.block) * g.block * g.channels * sizeof(float);
    if (bytes > kSharedBudget) return false;
    spatial_match_select_kernel<<<ref_count, kSelectThreads, bytes, stream>>>(plane, g, grid, ref_begin, out, counts,
                                                                              candidates);
    return true;
}

template <int N>
void launch_spatial(const float* plane, const MatchGeometry& g, const RasterGrid& grid, int ref_begin, int ref_count,
                    DeviceMatch* out, int* counts, cudaStream_t stream) {
    const int span = g.lo() + g.hi();
    const std::size_t window = static_cast<std::size_t>(span + g.block) * (span + g.block) * g.channels * sizeof(float);
    const bool staged = window + N * sizeof(MatchKey) + BlockTopK<N>::kSortBytes <= kSharedBudget;
    spatial_match_kernel<N><<<ref_count, kThreads, staged ? window : 0, stream>>>(plane, g, grid, ref_begin, out,
                                                                                    counts, staged);
}

// ---------------------------------------------------------------------------
// Tensor-core matcher of large windows (spatial_match_tensor). The distance of
// a reference r and a candidate c is first taken as d' = |c'|^2 + |r'|^2 -
// 2 r'.c' over FP16 values v' = fp16(v - mu), mu the plane's mean (FP32 accumulation, one
// mma.m16n8k16 per 16 candidates x 8 references x 16 pixels). Rounding bounds
// |sqrt(d) - sqrt(d')| by e_r + e_c, e = ||v - mu - v'|| over a patch (the
// triangle inequality), up to the FP32 error of d', at most kTensorUlps ulps of
// |r'|^2 + |c'|^2 (10 measured over 8.7M pairs). Two passes:
//   0. the K smallest d' over the candidates within kTensorSubset of the
//      references (the K-th of any subset bounds the window's K-th from
//      above): U, an upper bound of the K-th true distance;
//   1. every candidate of the window that d' does not place above U;
// then the survivors' exact keys in spatial_match_lane_kernel's arithmetic
// and order: the lane kernels' matches, bit for bit. A reference whose
// survivors overflow kTensorList takes every candidate exactly, and so does
// every reference when the plane holds a value FP16 cannot (scratch flag).
// TWSC's default shape (window 60, group 8) 2.0-2.1x the lane kernel at
// 1080p; the side of 33 is where it stops paying (RTX 5080, four images, sigma
// 3 and 25; docs/cuda-head-r2-20261020-tools/t48_subset.cu).
constexpr int kTensorPad = 112;   // pair columns beyond each side of the plane
constexpr int kTensorList = 256;  // survivors a reference keeps
constexpr int kTensorSubset = 8;
constexpr int kTensorMinSide = 33, kTensorMaxSide = 129;
constexpr float kTensorEps = 5.9604645e-8f, kTensorUlps = 128.f;
constexpr unsigned kTensorInfBits = 0x7f800000u;
// Header: [0] emax (float bits, max), [1] the plane is unusable, [2] mu (float bits); then a double per row
// (the mean's row sums: a fixed order, so the workspace is the same on every run).
constexpr std::size_t kTensorHeaderFixed = 256;
__host__ __device__ inline std::size_t tensor_header(int height) {
    return kTensorHeaderFixed + ((static_cast<std::size_t>(height) * sizeof(double) + 255) & ~std::size_t{255});
}

struct TensorPlane {
    const unsigned* pairs;  // height rows of pitch pairs: (v'(x), v'(x + 1)), column x at kTensorPad + x
    const float* norms;     // |c'|^2 of the patch at (y, x), width x height
    const float* errs;      // ||c - mu - c'|| of that patch
    const unsigned* header;
    int pitch;
};

__host__ __device__ inline int tensor_pitch(int width) { return width + 2 * kTensorPad; }

TensorPlane tensor_plane(const void* scratch, int width, int height) {
    auto* bytes = static_cast<const unsigned char*>(scratch);
    TensorPlane t;
    t.header = reinterpret_cast<const unsigned*>(bytes);
    t.pitch = tensor_pitch(width);
    t.pairs = reinterpret_cast<const unsigned*>(bytes + tensor_header(height));
    t.norms = reinterpret_cast<const float*>(t.pairs + static_cast<std::size_t>(t.pitch) * height);
    t.errs = t.norms + static_cast<std::size_t>(width) * height;
    return t;
}

__device__ __forceinline__ unsigned short tensor_half(float v, float mu) { return __half_as_ushort(__float2half_rn(v - mu)); }

// The plane's mean: each block a row (a fixed strided order and tree), then one block over the rows.
constexpr int kTensorMeanThreads = 256;
__device__ __forceinline__ double tensor_block_sum(double v) {
    __shared__ double part[kTensorMeanThreads / 32];
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = v;
    __syncthreads();
    double total = 0;
    if (threadIdx.x == 0) {
        for (int w = 0; w < kTensorMeanThreads / 32; ++w) total += part[w];
    }
    return total;
}

__global__ void __launch_bounds__(kTensorMeanThreads) tensor_rows_kernel(const float* plane, MatchGeometry g, unsigned* header) {
    const float* row = plane + static_cast<long long>(blockIdx.x) * g.pitch;
    double v = 0;
    for (int x = threadIdx.x; x < g.width; x += kTensorMeanThreads) v += row[x];
    const double total = tensor_block_sum(v);
    if (threadIdx.x == 0) reinterpret_cast<double*>(reinterpret_cast<unsigned char*>(header) + kTensorHeaderFixed)[blockIdx.x] = total;
}

__global__ void __launch_bounds__(kTensorMeanThreads) tensor_mean_kernel(MatchGeometry g, unsigned* header) {
    const double* rows = reinterpret_cast<const double*>(reinterpret_cast<const unsigned char*>(header) + kTensorHeaderFixed);
    double v = 0;
    for (int y = threadIdx.x; y < g.height; y += kTensorMeanThreads) v += rows[y];
    const double total = tensor_block_sum(v);
    if (threadIdx.x == 0) header[2] = __float_as_uint(static_cast<float>(total / (static_cast<double>(g.width) * g.height)));
}

// Pairs of every column of the padded plane; values beyond the plane are mu.
__global__ void tensor_pairs_kernel(const float* plane, MatchGeometry g, unsigned* pairs, unsigned* header) {
    const float mu = __uint_as_float(header[2]);
    const int pitch = tensor_pitch(g.width);
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<long long>(pitch) * g.height) return;
    const int y = static_cast<int>(i / pitch), x = static_cast<int>(i % pitch) - kTensorPad;
    const float* row = plane + static_cast<long long>(y) * g.pitch;
    const float a = x >= 0 && x < g.width ? row[x] : mu;
    const float b = x + 1 >= 0 && x + 1 < g.width ? row[x + 1] : mu;
    if (x >= 0 && x < g.width && !(fabsf(a - mu) <= 60000.f)) atomicOr(header + 1, 1u);  // NaN mu fails too
    pairs[i] = static_cast<unsigned>(tensor_half(a, mu)) | static_cast<unsigned>(tensor_half(b, mu)) << 16;
}

// |c'|^2 and ||c - mu - c'|| of every patch position, and their largest error.
constexpr int kTensorTileX = 32, kTensorTileY = 8;
__global__ void __launch_bounds__(kTensorTileX * kTensorTileY) tensor_norms_kernel(const float* plane, MatchGeometry g,
                                                                                  float* norms, float* errs,
                                                                                  unsigned* header) {
    __shared__ float q[kTensorTileY + 7][kTensorTileX + 8], r[kTensorTileY + 7][kTensorTileX + 8];
    const float mu = __uint_as_float(header[2]);
    const int x0 = blockIdx.x * kTensorTileX, y0 = blockIdx.y * kTensorTileY;
    for (int e = threadIdx.x; e < (kTensorTileY + 7) * (kTensorTileX + 7); e += blockDim.x) {
        const int yy = e / (kTensorTileX + 7), xx = e % (kTensorTileX + 7);
        const int y = min(y0 + yy, g.height - 1), x = min(x0 + xx, g.width - 1);
        const float v = plane[static_cast<long long>(y) * g.pitch + x] - mu;
        const float h = __half2float(__float2half_rn(v));
        q[yy][xx] = h;
        r[yy][xx] = v - h;
    }
    __syncthreads();
    const int tx = threadIdx.x % kTensorTileX, ty = threadIdx.x / kTensorTileX;
    const int x = x0 + tx, y = y0 + ty;
    if (x > g.width - 8 || y > g.height - 8) return;
    float n = 0.f, e2 = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            n = fmaf(q[ty + i][tx + j], q[ty + i][tx + j], n);
            e2 = fmaf(r[ty + i][tx + j], r[ty + i][tx + j], e2);
        }
    }
    const float err = sqrtf(e2) * 1.001f + 1e-9f;  // above the true norm despite FP32 rounding
    const long long at = static_cast<long long>(y) * g.width + x;
    norms[at] = n;
    errs[at] = err;
    atomicMax(header, __float_as_uint(err));
}

template <int N>
__device__ __forceinline__ void tensor_insert(float v, float (&s)[N]) {
    if (!(v < s[N - 1])) return;
    s[N - 1] = v;
#pragma unroll
    for (int i = N - 1; i > 0; --i) {
        if (s[i] < s[i - 1]) {
            const float t = s[i];
            s[i] = s[i - 1];
            s[i - 1] = t;
        }
    }
}

// The exact key of candidate (cx, cy) for the reference at (rx, ry): the lane
// kernels' sum (per column over the rows, then the columns' tree) and order.
__device__ __forceinline__ unsigned long long tensor_key(const float* plane, int pitch, int rx, int ry, int cx, int cy) {
    float s[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        float sum = 0.f;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const float d = plane[static_cast<long long>(ry + k) * pitch + rx + j] -
                            plane[static_cast<long long>(cy + k) * pitch + cx + j];
            sum = fmaf(d, d, sum);
        }
        s[j] = sum;
    }
    const float t = ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
    return static_cast<unsigned long long>(min(__float_as_uint(t), kTensorInfBits)) << 32 |
           static_cast<unsigned>(cy) << 16 | static_cast<unsigned>(cx);
}

// U's root plus e_r from an upper bound sk of a reference's K-th d', and the
// limit with the plane's largest candidate error (a superset for pass 1).
__device__ __forceinline__ void tensor_bounds(float sk, float nr, float er, float emax, float& thr, float& wide) {
    if (sk >= 1e30f) {
        thr = INFINITY, wide = INFINITY;
        return;
    }
    const float delta = kTensorUlps * kTensorEps * (3.f * nr + 2.f * sk + 1e-3f);  // |c'|^2 <= 2 |r'|^2 + 2 d'
    const float root = sqrtf(sk + delta) + er + emax;
    thr = root * (1.f + 4e-6f) + er;
    const float widest = thr + emax;
    wide = widest * widest * (1.f + 4e-6f);
}

struct TensorLists {
    unsigned short at[8][kTensorList];  // window position (dy + lo) * side + dx + lo
    int n[8];
};

// A warp takes 8 consecutive references of the batch (at most two grid rows,
// grid.nx >= 8); a lane holds the 16 x 8 tile outputs of its group / thread in
// group: candidates g and g + 8 of the tile, references 2 t and 2 t + 1.
template <int KMAX>
__global__ void __launch_bounds__(128, 4) tensor_match_kernel(const float* plane, MatchGeometry g, RasterGrid grid,
                                                              int ref_begin, int ref_count, DeviceMatch* out,
                                                              int* counts, TensorPlane tp) {
    __shared__ TensorLists lists[4];
    TensorLists& L = lists[threadIdx.x >> 5];
    const int lane = threadIdx.x & 31;
    const int first = (blockIdx.x * 4 + (threadIdx.x >> 5)) * 8;
    if (first >= ref_count) return;
    const int nref = min(8, ref_count - first);
    const int k = min(g.group, kMaxGroup) - 1;
    const int lo = g.lo(), hi = g.hi(), side = lo + hi + 1;
    const int max_x = g.width - 8, max_y = g.height - 8;
    const int gq = lane >> 2, tq = lane & 3;
    const float emax = __uint_as_float(tp.header[0]) * 1.001f + 1e-9f;
    const bool usable = tp.header[1] == 0;
    if (lane < 8) L.n[lane] = usable ? 0 : kTensorList + 1;
    __syncwarp();
    const auto ref_x = [&](int j) { return grid.x(ref_begin + first + j); };
    const auto ref_y = [&](int j) { return grid.y(ref_begin + first + j); };
    const int row0 = ref_y(0);
    int split = nref;
    for (int j = 1; j < nref; ++j) {
        if (ref_y(j) != row0) {
            split = j;
            break;
        }
    }
    for (int part = 0; usable && part < (split < nref ? 2 : 1); ++part) {
        const int jlo = part ? split : 0, jhi = part ? nref : split;
        const int yr = ref_y(jlo);
        const int xmin = ref_x(jlo), xmax = ref_x(jhi - 1);
        const int xg = ref_x(min(max(gq, jlo), jhi - 1));
        unsigned b[4][2];
        const unsigned* bcol = tp.pairs + static_cast<long long>(yr) * tp.pitch + kTensorPad + xg + 2 * tq;
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            b[ks][0] = bcol[(2 * ks) * tp.pitch];
            b[ks][1] = bcol[(2 * ks + 1) * tp.pitch];
        }
        float nr[2], er[2];
        int xr[2];
        bool act[2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int j = 2 * tq + e;
            act[e] = j >= jlo && j < jhi;
            xr[e] = ref_x(min(max(j, jlo), jhi - 1));
            nr[e] = tp.norms[static_cast<long long>(yr) * g.width + xr[e]];
            er[e] = tp.errs[static_cast<long long>(yr) * g.width + xr[e]];
        }
        const int x_lo = xmin - lo;
        const int tiles = (xmax + hi - x_lo) / 16 + 1;  // <= 16 (spatial_match_tensor)
        // Window and plane membership of the lane's outputs per tile (bit 4 tile + 2 h + e), and the
        // candidate at the reference's own position (cleared on its row).
        unsigned long long cols = 0, self = 0;
        for (int tile = 0; tile < tiles; ++tile) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int xc = x_lo + 16 * tile + gq + 8 * h;
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int dx = xc - xr[e];
                    const unsigned long long bit = 1ull << (4 * tile + 2 * h + e);
                    if (act[e] && xc >= 0 && xc <= max_x && dx >= -lo && dx <= hi) cols |= bit;
                    if (dx == 0) self |= bit;
                }
            }
        }
        const int y0 = max(yr - lo, 0), y1 = min(yr + hi, max_y);
        const int t0 = max(0, (xmin - min(kTensorSubset, lo) - x_lo) / 16);
        const int t1 = min(tiles, (xmax + min(kTensorSubset, hi) - x_lo) / 16 + 1);
        // The KMAX smallest d' per reference, the first KMAX - k slots held by -1 (d' >= 0): the last slot is
        // the k-th smallest without an index known only at run time (that would put the lists in local memory).
        float best[2][KMAX];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
#pragma unroll
            for (int i = 0; i < KMAX; ++i) best[e][i] = i < KMAX - k ? -1.f : 1e30f;
        }
        float thr[2] = {0.f, 0.f}, wide[2] = {0.f, 0.f};
        for (int pass = 0; pass < 2; ++pass) {
            const int ya = pass ? y0 : max(y0, yr - kTensorSubset), yb = pass ? y1 : min(y1, yr + kTensorSubset);
            const int ta = pass ? 0 : t0, tb = pass ? tiles : t1;
            for (int y = ya; y <= yb; ++y) {
                const unsigned long long mask = y == yr ? cols & ~self : cols;
                const unsigned* q = tp.pairs + static_cast<long long>(y) * tp.pitch + kTensorPad + x_lo + gq + 2 * tq + 16 * ta;
                const float* nq = tp.norms + static_cast<long long>(y) * g.width + x_lo + gq;
                const float* eq = tp.errs + static_cast<long long>(y) * g.width + x_lo + gq;
                for (int tile = ta; tile < tb; ++tile, q += 16) {
                    float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                    for (int ks = 0; ks < 4; ++ks) {
                        const unsigned* r0 = q + (2 * ks) * tp.pitch;
                        const unsigned* r1 = r0 + tp.pitch;
                        const unsigned a0 = __ldg(r0), a1 = __ldg(r0 + 8), a2 = __ldg(r1), a3 = __ldg(r1 + 8);
#if __CUDA_ARCH__ >= 800
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                            "{%0,%1,%2,%3};"
                            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b[ks][0]), "r"(b[ks][1]));
#else
                        (void)a0, (void)a1, (void)a2, (void)a3;
                        __trap();
#endif
                    }
                    const unsigned bits = static_cast<unsigned>(mask >> (4 * tile)) & 15u;
                    if (!bits) continue;
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        if (!(bits >> (2 * h) & 3u)) continue;
                        const float nc = __ldg(nq + 16 * tile + 8 * h);
#pragma unroll
                        for (int e = 0; e < 2; ++e) {
                            if (!(bits >> (2 * h + e) & 1u)) continue;
                            const float d = fmaxf(fmaf(-2.f, c[2 * h + e], nc + nr[e]), 0.f);
                            if (pass == 0) {
                                tensor_insert(d, best[e]);
                            } else {
                                const float slack = kTensorUlps * kTensorEps * (nr[e] + nc);
                                if (!(d <= wide[e] + slack)) continue;
                                const float lim = thr[e] + __ldg(eq + 16 * tile + 8 * h);
                                if (d <= lim * lim * (1.f + 4e-6f) + slack) {
                                    const int slot = atomicAdd(&L.n[2 * tq + e], 1);
                                    const int xc = x_lo + 16 * tile + gq + 8 * h;
                                    if (slot < kTensorList) {
                                        L.at[2 * tq + e][slot] =
                                            static_cast<unsigned short>((y - yr + lo) * side + xc - xr[e] + lo);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if (pass == 0) {
                // The lists of the 8 lanes of each reference (lane bits 2 to 4), then the bounds.
#pragma unroll
                for (int o = 4; o < 32; o <<= 1) {
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        float v[KMAX];
#pragma unroll
                        for (int i = 0; i < KMAX; ++i) v[i] = __shfl_xor_sync(~0u, best[e][i], o);
#pragma unroll
                        for (int i = 0; i < KMAX; ++i) {
                            if (v[i] >= 0.f) tensor_insert(v[i], best[e]);  // the partner's own -1 stay out
                        }
                    }
                }
#pragma unroll
                for (int e = 0; e < 2; ++e) tensor_bounds(best[e][KMAX - 1], nr[e], er[e], emax, thr[e], wide[e]);
            }
        }
    }
    __syncwarp();
    // Exact re-rank: the k smallest keys among each reference's survivors.
    for (int j = 0; j < nref; ++j) {
        const int r = first + j, rx = ref_x(j), ry = ref_y(j);
        const int count = L.n[j];
        unsigned long long held[KMAX];
#pragma unroll
        for (int i = 0; i < KMAX; ++i) held[i] = ~0ull;
        const auto take = [&](int wi) {
            const int cy = ry - lo + wi / side, cx = rx - lo + wi % side;
            if (cy < 0 || cy > max_y || cx < 0 || cx > max_x || (cx == rx && cy == ry)) return;
            const unsigned long long key = tensor_key(plane, g.pitch, rx, ry, cx, cy);
            if (key < held[KMAX - 1]) {
                held[KMAX - 1] = key;
#pragma unroll
                for (int i = KMAX - 1; i > 0; --i) {
                    if (held[i] < held[i - 1]) {
                        const unsigned long long s = held[i];
                        held[i] = held[i - 1];
                        held[i - 1] = s;
                    }
                }
            }
        };
        if (count <= kTensorList) {
            for (int i = lane; i < count; i += 32) take(L.at[j][i]);
        } else {
            for (int wi = lane; wi < side * side; wi += 32) take(wi);
        }
        DeviceMatch* dst = out + static_cast<long long>(r) * g.group;
        int filled = 0;
        for (int m = 0; m < k; ++m) {
            unsigned long long v = held[0];
#pragma unroll
            for (int o = 16; o; o >>= 1) {
                const unsigned long long w = __shfl_xor_sync(~0u, v, o);
                v = w < v ? w : v;
            }
            if (v == ~0ull) break;
            if (held[0] == v) {  // keys are unique: one lane pops
#pragma unroll
                for (int i = 0; i < KMAX - 1; ++i) held[i] = held[i + 1];
                held[KMAX - 1] = ~0ull;
            }
            if (lane == 0) {
                const unsigned at = static_cast<unsigned>(v);
                dst[1 + m] = DeviceMatch{static_cast<int>(at & 0xffffu), static_cast<int>(at >> 16), 0,
                                         __uint_as_float(static_cast<unsigned>(v >> 32))};
            }
            ++filled;
        }
        if (lane == 0) {
            dst[0] = DeviceMatch{rx, ry, 0, 0.f};
            counts[r] = 1 + filled;
        }
    }
}

// mma.m16n8k16 on FP16 needs sm_80+ code, so the kernel this device runs
// must come from compute_80 or later: a GPU without a cubin of its own (or
// with PTX JIT forced) runs the compute_75 PTX, where the kernel traps.
// Cached per device.
bool tensor_device() {
    int device = 0;
    NSS_CUDA_CHECK(cudaGetDevice(&device));
    static std::array<std::atomic<int>, 64> known{};  // 0 unknown, 1 no, 2 yes
    if (device < 64 && known[device]) return known[device] == 2;
    // A device that cannot load the kernel at all keeps the lane kernels (they report their own errors).
    cudaFuncAttributes attributes{};
    const bool loaded = cudaFuncGetAttributes(&attributes, tensor_match_kernel<7>) == cudaSuccess;
    if (!loaded) (void)cudaGetLastError();
    const bool yes = loaded && attributes.ptxVersion >= 80;
    if (device < 64) known[device] = yes ? 2 : 1;
    return yes;
}

bool launch_tensor(const float* plane, const MatchGeometry& g, const RasterGrid& grid, int ref_begin, int ref_count,
                   DeviceMatch* out, int* counts, const void* scratch, cudaStream_t stream) {
    if (!scratch || !spatial_match_tensor(g, grid)) return false;
    const TensorPlane tp = tensor_plane(scratch, g.width, g.height);
    const unsigned blocks = static_cast<unsigned>((ref_count + 31) / 32);
    if (g.group <= 8) {
        tensor_match_kernel<7><<<blocks, 128, 0, stream>>>(plane, g, grid, ref_begin, ref_count, out, counts, tp);
    } else {
        tensor_match_kernel<15><<<blocks, 128, 0, stream>>>(plane, g, grid, ref_begin, ref_count, out, counts, tp);
    }
    NSS_CUDA_CHECK_LAUNCH();
    return true;
}

}  // namespace

RasterGrid make_raster_grid(int width, int height, int block, int step) {
    RasterGrid grid{0, 0, step, width - block, height - block};
    if (width < block || height < block || step < 1) return grid;
    grid.nx = (width - block + step - 1) / step + 1;
    grid.ny = (height - block + step - 1) / step + 1;
    return grid;
}

bool spatial_match_tensor(const MatchGeometry& g, const RasterGrid& grid) {
    const int side = g.lo() + g.hi() + 1;
    return g.channels == 1 && g.block == 8 && g.group >= 2 && g.group <= 16 && side >= kTensorMinSide &&
           side <= kTensorMaxSide && grid.nx >= 8 && (7 * grid.step + g.lo() + g.hi()) / 16 + 1 <= 16 &&
           g.width <= kLaneMaxCoordinate && g.height <= kLaneMaxCoordinate && g.width >= 8 && g.height >= 8 &&
           tensor_device();
}

std::size_t spatial_match_scratch_bytes(int width, int height) {
    return tensor_header(height) + static_cast<std::size_t>(tensor_pitch(width)) * height * sizeof(unsigned) +
           2 * static_cast<std::size_t>(width) * height * sizeof(float);
}

void spatial_match_prepare(const float* plane, const MatchGeometry& g, void* scratch, cudaStream_t stream) {
    const TensorPlane tp = tensor_plane(scratch, g.width, g.height);
    auto* header = const_cast<unsigned*>(tp.header);
    NSS_CUDA_CHECK(cudaMemsetAsync(header, 0, 2 * sizeof(unsigned), stream));
    tensor_rows_kernel<<<static_cast<unsigned>(g.height), kTensorMeanThreads, 0, stream>>>(plane, g, header);
    NSS_CUDA_CHECK_LAUNCH();
    tensor_mean_kernel<<<1, kTensorMeanThreads, 0, stream>>>(g, header);
    NSS_CUDA_CHECK_LAUNCH();
    const long long cells = static_cast<long long>(tp.pitch) * g.height;
    tensor_pairs_kernel<<<static_cast<unsigned>((cells + 255) / 256), 256, 0, stream>>>(plane, g,
                                                                                         const_cast<unsigned*>(tp.pairs), header);
    NSS_CUDA_CHECK_LAUNCH();
    const dim3 tiles((g.width + kTensorTileX - 1) / kTensorTileX, (g.height + kTensorTileY - 1) / kTensorTileY);
    tensor_norms_kernel<<<tiles, kTensorTileX * kTensorTileY, 0, stream>>>(plane, g, const_cast<float*>(tp.norms),
                                                                          const_cast<float*>(tp.errs), header);
    NSS_CUDA_CHECK_LAUNCH();
}

void spatial_match(const float* plane, const MatchGeometry& geometry, const RasterGrid& grid, int ref_begin,
                   int ref_count, DeviceMatch* out, int* counts, cudaStream_t stream, const void* scratch) {
    if (ref_count <= 0) return;
    if (launch_tensor(plane, geometry, grid, ref_begin, ref_count, out, counts, scratch, stream)) return;
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
    if (launch_select(plane, geometry, grid, ref_begin, ref_count, out, counts, stream)) {
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
