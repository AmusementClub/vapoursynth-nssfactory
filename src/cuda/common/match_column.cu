// SPDX-License-Identifier: GPL-2.0-only
// Exact column-reuse matcher for BM3D block 4/8/12/16 searches at steps up to three.
#include "cuda/common/match.hpp"
#include "cuda/runtime/error.hpp"

namespace nss_cuda {
namespace {

constexpr int kThreads = 128;
constexpr int kQueue = 8;
constexpr unsigned kInf = 0x7f800000u;
constexpr unsigned long long kEmpty = ~0ull;
constexpr unsigned kAll = 0xffffffffu;

template <int K>
__device__ __forceinline__ void table_insert(unsigned long long (&table)[K], unsigned long long key) {
#pragma unroll
    for (int i = K - 1; i > 0; --i) {
        table[i] = key < table[i - 1] ? table[i - 1] : (key < table[i] ? key : table[i]);
    }
    table[0] = key < table[0] ? key : table[0];
}

template <int K>
__device__ __forceinline__ void flush(unsigned long long (&table)[K],
                                      unsigned long long (*queue)[kThreads], int& queued) {
#pragma unroll
    for (int q = 0; q < kQueue; ++q) {
        if (q < queued) table_insert<K>(table, queue[q][threadIdx.x]);
    }
    queued = 0;
}

__device__ __forceinline__ float tree4(const float* values) {
    return (values[0] + values[1]) + (values[2] + values[3]);
}

__device__ __forceinline__ float tree8(const float* values) {
    return tree4(values) + tree4(values + 4);
}

template <int B>
__device__ __forceinline__ float column_total(const float* values);
template <>
__device__ __forceinline__ float column_total<4>(const float* values) {
    return tree4(values);
}
template <>
__device__ __forceinline__ float column_total<8>(const float* values) {
    return tree8(values);
}
template <>
__device__ __forceinline__ float column_total<12>(const float* values) {
    // The lane kernel's padded 16-lane tree: lanes 12-15 contribute exact zero.
    return tree8(values) + tree4(values + 8);
}
template <>
__device__ __forceinline__ float column_total<16>(const float* values) {
    return tree8(values) + tree8(values + 8);
}

struct RowBlock {
    int row;
    int lo_col;
    int hi_col;
    bool any;
};

__device__ __forceinline__ RowBlock row_block(const RasterGrid& grid, int row, int col_first, int ref_begin,
                                              int ref_count) {
    RowBlock block;
    block.row = row;
    block.lo_col = max(col_first, ref_begin - row * grid.nx);
    block.hi_col = min(min(col_first + kThreads, grid.nx), ref_begin + ref_count - row * grid.nx) - 1;
    block.any = row < grid.ny && block.lo_col <= block.hi_col;
    return block;
}

template <int K>
__device__ __forceinline__ void write_out(const unsigned long long (&table)[K], int cx, int cy, int ref, int group,
                                          DeviceMatch* out, int* counts) {
    DeviceMatch* dst = out + static_cast<long long>(ref) * group;
    dst[0] = DeviceMatch{cx, cy, 0, 0.f};
    int filled = 0;
#pragma unroll
    for (int i = 0; i < K; ++i) {
        if (table[i] != kEmpty) {
            const unsigned at = static_cast<unsigned>(table[i]);
            const unsigned bits = static_cast<unsigned>(table[i] >> 32);
            dst[1 + i] = DeviceMatch{static_cast<int>(at & 0xffffu), static_cast<int>(at >> 16), 0,
                                     __uint_as_float(bits)};
            ++filled;
        }
    }
    counts[ref] = 1 + filled;
}

template <int B, int K, int CPT>
__global__ void __launch_bounds__(kThreads) column_match_kernel(const float* plane, MatchGeometry g, RasterGrid grid,
                                                               int ref_begin, int ref_count, int row_first, int segs,
                                                               DeviceMatch* out, int* counts) {
    __shared__ float colsum[2][kThreads * CPT + B];
    __shared__ unsigned long long queue[kQueue][kThreads];
    const int tid = threadIdx.x;
    const int col_first = (blockIdx.x % segs) * kThreads;
    const RowBlock block = row_block(grid, row_first + blockIdx.x / segs, col_first, ref_begin, ref_count);
    if (!block.any) return;
    const int base = block.row * grid.nx;
    const int col = col_first + tid;
    const bool live = col >= block.lo_col && col <= block.hi_col;
    const int x0 = grid.x(base + block.lo_col);
    const int span = grid.x(base + block.hi_col) - x0 + B;
    const int yr = grid.y(base + block.lo_col);
    const int cx = live ? grid.x(base + col) : x0;
    const int lo = g.lo(), hi = g.hi();
    const int max_x = g.width - B, max_y = g.height - B;
    const int last_row = g.height - 1, last_col = g.width - 1;
    const long long pitch = g.pitch;

    float ref[CPT][B];
    bool own[CPT];
#pragma unroll
    for (int c = 0; c < CPT; ++c) {
        const int j = tid + c * kThreads;
        own[c] = j < span;
        const int x = min(x0 + j, last_col);
#pragma unroll
        for (int k = 0; k < B; ++k) ref[c][k] = plane[(yr + k) * pitch + x];
    }
    unsigned long long table[K];
#pragma unroll
    for (int i = 0; i < K; ++i) table[i] = kEmpty;
    int queued = 0;
    int buf = 0;

    for (int dx = -lo; dx <= hi; ++dx) {
        int xs[CPT];
        float values[CPT][B];
#pragma unroll
        for (int c = 0; c < CPT; ++c) xs[c] = min(max(x0 + tid + c * kThreads + dx, 0), last_col);
#pragma unroll
        for (int k = 0; k < B - 1; ++k) {
            const long long row = min(max(yr - lo + k, 0), last_row) * pitch;
#pragma unroll
            for (int c = 0; c < CPT; ++c) values[c][k + 1] = plane[row + xs[c]];
        }
        const int xc = cx + dx;
        const bool col_ok = live && xc >= 0 && xc <= max_x;
        int dy = -lo;
        while (dy <= hi) {
#pragma unroll
            for (int s = 0; s < B; ++s) {
                if (dy > hi) break;
                const int ytop = yr + dy;
                const long long row = min(max(ytop + B - 1, 0), last_row) * pitch;
#pragma unroll
                for (int c = 0; c < CPT; ++c) {
                    values[c][s] = plane[row + xs[c]];
                    if (own[c]) {
                        float sum = 0.f;
#pragma unroll
                        for (int k = 0; k < B; ++k) {
                            const float d = ref[c][k] - values[c][(s + 1 + k) % B];
                            sum = fmaf(d, d, sum);
                        }
                        colsum[buf][tid + c * kThreads] = sum;
                    }
                }
                __syncthreads();
                if (col_ok && ytop >= 0 && ytop <= max_y && (dx | dy) != 0) {
                    const float sum = column_total<B>(colsum[buf] + (cx - x0));
                    const unsigned long long key =
                        static_cast<unsigned long long>(min(__float_as_uint(sum), kInf)) << 32 |
                        static_cast<unsigned>(ytop) << 16 | static_cast<unsigned>(xc);
                    if (key < table[K - 1]) queue[queued++][tid] = key;
                }
                if (__any_sync(kAll, queued == kQueue)) flush<K>(table, queue, queued);
                buf ^= 1;
                ++dy;
            }
        }
    }
    flush<K>(table, queue, queued);
    if (live) write_out<K>(table, cx, yr, base + col - ref_begin, g.group, out, counts);
}

// With eight lanes and blocks wider than eight, a lane sums its two columns
// row by row before the xor tree. Reuse those paired sums; block 12 has four
// single-column lanes after the four paired lanes.
template <int B, int CPT>
__global__ void __launch_bounds__(kThreads) column_match_pair8_kernel(const float* plane, MatchGeometry g, RasterGrid grid,
                                                                     int ref_begin, int ref_count, int row_first, int segs,
                                                                     DeviceMatch* out, int* counts) {
    static_assert(B == 12 || B == 16);
    constexpr int kSingleColumns = B == 12 ? kThreads * CPT + B : 1;
    __shared__ float single[2][kSingleColumns];
    __shared__ float paired[2][kThreads * CPT + B];
    __shared__ unsigned long long queue[kQueue][kThreads];
    const int tid = threadIdx.x;
    const int col_first = (blockIdx.x % segs) * kThreads;
    const RowBlock block = row_block(grid, row_first + blockIdx.x / segs, col_first, ref_begin, ref_count);
    if (!block.any) return;
    const int base = block.row * grid.nx;
    const int col = col_first + tid;
    const bool live = col >= block.lo_col && col <= block.hi_col;
    const int x0 = grid.x(base + block.lo_col);
    const int span = grid.x(base + block.hi_col) - x0 + B;
    const int yr = grid.y(base + block.lo_col);
    const int cx = live ? grid.x(base + col) : x0;
    const int lo = g.lo(), hi = g.hi();
    const int max_x = g.width - B, max_y = g.height - B;
    const int last_row = g.height - 1, last_col = g.width - 1;
    const long long pitch = g.pitch;

    float ref[CPT][B], ref_hi[CPT][B];
    bool own[CPT];
#pragma unroll
    for (int c = 0; c < CPT; ++c) {
        const int j = tid + c * kThreads;
        own[c] = j < span;
        const int x = min(x0 + j, last_col), x_hi = min(x0 + j + 8, last_col);
#pragma unroll
        for (int k = 0; k < B; ++k) {
            ref[c][k] = plane[(yr + k) * pitch + x];
            ref_hi[c][k] = plane[(yr + k) * pitch + x_hi];
        }
    }
    unsigned long long table[7];
#pragma unroll
    for (int i = 0; i < 7; ++i) table[i] = kEmpty;
    int queued = 0;
    int buf = 0;

    for (int dx = -lo; dx <= hi; ++dx) {
        int xs[CPT], xs_hi[CPT];
        float values[CPT][B], values_hi[CPT][B];
#pragma unroll
        for (int c = 0; c < CPT; ++c) {
            xs[c] = min(max(x0 + tid + c * kThreads + dx, 0), last_col);
            xs_hi[c] = min(xs[c] + 8, last_col);
        }
#pragma unroll
        for (int k = 0; k < B - 1; ++k) {
            const long long row = min(max(yr - lo + k, 0), last_row) * pitch;
#pragma unroll
            for (int c = 0; c < CPT; ++c) {
                values[c][k + 1] = plane[row + xs[c]];
                values_hi[c][k + 1] = plane[row + xs_hi[c]];
            }
        }
        const int xc = cx + dx;
        const bool col_ok = live && xc >= 0 && xc <= max_x;
        int dy = -lo;
        while (dy <= hi) {
#pragma unroll
            for (int s = 0; s < B; ++s) {
                if (dy > hi) break;
                const int ytop = yr + dy;
                const long long row = min(max(ytop + B - 1, 0), last_row) * pitch;
#pragma unroll
                for (int c = 0; c < CPT; ++c) {
                    values[c][s] = plane[row + xs[c]];
                    values_hi[c][s] = plane[row + xs_hi[c]];
                    if (own[c]) {
                        float sum = 0.f, sum_hi = 0.f;
#pragma unroll
                        for (int k = 0; k < B; ++k) {
                            const float d = ref[c][k] - values[c][(s + 1 + k) % B];
                            if constexpr (B == 12) sum = fmaf(d, d, sum);
                            sum_hi = fmaf(d, d, sum_hi);
                            const float d_hi = ref_hi[c][k] - values_hi[c][(s + 1 + k) % B];
                            sum_hi = fmaf(d_hi, d_hi, sum_hi);
                        }
                        if constexpr (B == 12) single[buf][tid + c * kThreads] = sum;
                        paired[buf][tid + c * kThreads] = sum_hi;
                    }
                }
                __syncthreads();
                if (col_ok && ytop >= 0 && ytop <= max_y && (dx | dy) != 0) {
                    const int r = cx - x0;
                    float sum;
                    if constexpr (B == 12) {
                        sum = tree4(paired[buf] + r) + tree4(single[buf] + r + 4);
                    } else {
                        sum = tree8(paired[buf] + r);
                    }
                    const unsigned long long key =
                        static_cast<unsigned long long>(min(__float_as_uint(sum), kInf)) << 32 |
                        static_cast<unsigned>(ytop) << 16 | static_cast<unsigned>(xc);
                    if (key < table[6]) queue[queued++][tid] = key;
                }
                if (__any_sync(kAll, queued == kQueue)) flush<7>(table, queue, queued);
                buf ^= 1;
                ++dy;
            }
        }
    }
    flush<7>(table, queue, queued);
    if (live) write_out<7>(table, cx, yr, base + col - ref_begin, g.group, out, counts);
}

template <int B>
bool launch_pair8(const float* plane, const MatchGeometry& g, const RasterGrid& grid, int ref_begin, int ref_count,
                  DeviceMatch* out, int* counts, cudaStream_t stream) {
    const int row_first = ref_begin / grid.nx;
    const int row_last = (ref_begin + ref_count - 1) / grid.nx;
    const int segs = (grid.nx + kThreads - 1) / kThreads;
    const int blocks = (row_last - row_first + 1) * segs;
    switch (grid.step) {
#define NSS_COLUMN_PAIR_STEP(S)                                                                                  \
    case S:                                                                                                      \
        if constexpr (B == 16 && S == 3) {                                                                       \
            return false;                                                                                        \
        } else {                                                                                                 \
            column_match_pair8_kernel<B, S + 1><<<blocks, kThreads, 0, stream>>>(plane, g, grid, ref_begin,      \
                                                                                ref_count, row_first, segs, out, \
                                                                                counts);                         \
        }                                                                                                        \
        break;
        NSS_COLUMN_PAIR_STEP(1)
        NSS_COLUMN_PAIR_STEP(2)
        NSS_COLUMN_PAIR_STEP(3)
#undef NSS_COLUMN_PAIR_STEP
    default: return false;
    }
    NSS_CUDA_CHECK_LAUNCH();
    return true;
}

template <int B, int K>
bool launch_column(const float* plane, const MatchGeometry& g, const RasterGrid& grid, int ref_begin, int ref_count,
                   DeviceMatch* out, int* counts, cudaStream_t stream) {
    const int row_first = ref_begin / grid.nx;
    const int row_last = (ref_begin + ref_count - 1) / grid.nx;
    const int segs = (grid.nx + kThreads - 1) / kThreads;
    const int blocks = (row_last - row_first + 1) * segs;
    switch (grid.step) {
#define NSS_COLUMN_STEP(S)                                                                                       \
    case S:                                                                                                      \
        if constexpr ((B == 4 && S == 2) || (B == 16 && K == 31)) {                                             \
            return false;                                                                                        \
        } else {                                                                                                 \
            column_match_kernel<B, K, S + 1><<<blocks, kThreads, 0, stream>>>(plane, g, grid, ref_begin,         \
                                                                              ref_count, row_first, segs, out,   \
                                                                              counts);                           \
        }                                                                                                        \
        break;
        NSS_COLUMN_STEP(1)
        NSS_COLUMN_STEP(2)
        NSS_COLUMN_STEP(3)
#undef NSS_COLUMN_STEP
    default: return false;
    }
    NSS_CUDA_CHECK_LAUNCH();
    return true;
}

}  // namespace

bool spatial_match_column_takes(const MatchGeometry& g, const RasterGrid& grid) {
    const bool group_ok = g.group == 8 || g.group == 16 || g.group == 32;
    const bool shape_ok = g.block == 8 || (g.block == 4 && grid.step != 2) || g.block == 12 ||
                          (g.block == 16 && g.group != 32 && !(g.group == 8 && grid.step == 3));
    return g.channels == 1 && shape_ok && group_ok && g.range_hi < 0 && g.width >= g.block && g.height >= g.block &&
           g.width <= 0xffff && g.height <= 0xffff && grid.nx >= 1 && grid.step >= 1 && grid.step <= 3;
}

bool spatial_match_column(const float* plane, const MatchGeometry& g, const RasterGrid& grid, int ref_begin,
                          int ref_count, DeviceMatch* out, int* counts, cudaStream_t stream) {
    if (ref_count <= 0 || !spatial_match_column_takes(g, grid)) return false;
#define NSS_COLUMN_BLOCK(B, K) return launch_column<B, K>(plane, g, grid, ref_begin, ref_count, out, counts, stream)
#define NSS_COLUMN_PAIR(B) return launch_pair8<B>(plane, g, grid, ref_begin, ref_count, out, counts, stream)
    switch (g.group) {
    case 8:
        switch (g.block) {
        case 4: NSS_COLUMN_BLOCK(4, 7);
        case 8: NSS_COLUMN_BLOCK(8, 7);
        case 12: NSS_COLUMN_PAIR(12);
        case 16: NSS_COLUMN_PAIR(16);
        default: return false;
        }
    case 16:
        switch (g.block) {
        case 4: NSS_COLUMN_BLOCK(4, 15);
        case 8: NSS_COLUMN_BLOCK(8, 15);
        case 12: NSS_COLUMN_BLOCK(12, 15);
        case 16: NSS_COLUMN_BLOCK(16, 15);
        default: return false;
        }
    case 32:
        switch (g.block) {
        case 4: NSS_COLUMN_BLOCK(4, 31);
        case 8: NSS_COLUMN_BLOCK(8, 31);
        case 12: NSS_COLUMN_BLOCK(12, 31);
        case 16: NSS_COLUMN_BLOCK(16, 31);
        default: return false;
        }
    default: return false;
    }
#undef NSS_COLUMN_BLOCK
#undef NSS_COLUMN_PAIR
}

}  // namespace nss_cuda
