// SPDX-License-Identifier: GPL-2.0-only
// Exact column-reuse matcher for BM3D block-8 searches at steps up to three.
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

__device__ __forceinline__ float tree8(const float* values) {
    return ((values[0] + values[1]) + (values[2] + values[3])) +
           ((values[4] + values[5]) + (values[6] + values[7]));
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

template <int K, int CPT>
__global__ void __launch_bounds__(kThreads) column_match_kernel(const float* plane, MatchGeometry g, RasterGrid grid,
                                                               int ref_begin, int ref_count, int row_first, int segs,
                                                               DeviceMatch* out, int* counts) {
    __shared__ float colsum[2][kThreads * CPT + 8];
    __shared__ unsigned long long queue[kQueue][kThreads];
    const int tid = threadIdx.x;
    const int col_first = (blockIdx.x % segs) * kThreads;
    const RowBlock block = row_block(grid, row_first + blockIdx.x / segs, col_first, ref_begin, ref_count);
    if (!block.any) return;
    const int base = block.row * grid.nx;
    const int col = col_first + tid;
    const bool live = col >= block.lo_col && col <= block.hi_col;
    const int x0 = grid.x(base + block.lo_col);
    const int span = grid.x(base + block.hi_col) - x0 + 8;
    const int yr = grid.y(base + block.lo_col);
    const int cx = live ? grid.x(base + col) : x0;
    const int lo = g.lo(), hi = g.hi();
    const int max_x = g.width - 8, max_y = g.height - 8;
    const int last_row = g.height - 1, last_col = g.width - 1;
    const long long pitch = g.pitch;

    float ref[CPT][8];
    bool own[CPT];
#pragma unroll
    for (int c = 0; c < CPT; ++c) {
        const int j = tid + c * kThreads;
        own[c] = j < span;
        const int x = min(x0 + j, last_col);
#pragma unroll
        for (int k = 0; k < 8; ++k) ref[c][k] = plane[(yr + k) * pitch + x];
    }
    unsigned long long table[K];
#pragma unroll
    for (int i = 0; i < K; ++i) table[i] = kEmpty;
    int queued = 0;
    int buf = 0;

    for (int dx = -lo; dx <= hi; ++dx) {
        int xs[CPT];
        float values[CPT][8];
#pragma unroll
        for (int c = 0; c < CPT; ++c) xs[c] = min(max(x0 + tid + c * kThreads + dx, 0), last_col);
#pragma unroll
        for (int k = 0; k < 7; ++k) {
            const long long row = min(max(yr - lo + k, 0), last_row) * pitch;
#pragma unroll
            for (int c = 0; c < CPT; ++c) values[c][k + 1] = plane[row + xs[c]];
        }
        const int xc = cx + dx;
        const bool col_ok = live && xc >= 0 && xc <= max_x;
        int dy = -lo;
        while (dy <= hi) {
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                if (dy > hi) break;
                const int ytop = yr + dy;
                const long long row = min(max(ytop + 7, 0), last_row) * pitch;
#pragma unroll
                for (int c = 0; c < CPT; ++c) {
                    values[c][s] = plane[row + xs[c]];
                    if (own[c]) {
                        float sum = 0.f;
#pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            const float d = ref[c][k] - values[c][(s + 1 + k) % 8];
                            sum = fmaf(d, d, sum);
                        }
                        colsum[buf][tid + c * kThreads] = sum;
                    }
                }
                __syncthreads();
                if (col_ok && ytop >= 0 && ytop <= max_y && (dx | dy) != 0) {
                    const float sum = tree8(colsum[buf] + (cx - x0));
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

template <int K>
bool launch_column(const float* plane, const MatchGeometry& g, const RasterGrid& grid, int ref_begin, int ref_count,
                   DeviceMatch* out, int* counts, cudaStream_t stream) {
    const int row_first = ref_begin / grid.nx;
    const int row_last = (ref_begin + ref_count - 1) / grid.nx;
    const int segs = (grid.nx + kThreads - 1) / kThreads;
    const int blocks = (row_last - row_first + 1) * segs;
    switch (grid.step) {
#define NSS_COLUMN_STEP(S)                                                                                       \
    case S:                                                                                                      \
        column_match_kernel<K, S + 1><<<blocks, kThreads, 0, stream>>>(plane, g, grid, ref_begin, ref_count,     \
                                                                       row_first, segs, out, counts);            \
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
    return g.channels == 1 && g.block == 8 && g.range_hi < 0 && g.width >= 8 && g.height >= 8 &&
           g.width <= 0xffff && g.height <= 0xffff && grid.nx >= 1 && grid.step >= 1 && grid.step <= 3 &&
           (g.group == 8 || g.group == 16 || g.group == 32);
}

bool spatial_match_column(const float* plane, const MatchGeometry& g, const RasterGrid& grid, int ref_begin,
                          int ref_count, DeviceMatch* out, int* counts, cudaStream_t stream) {
    if (ref_count <= 0 || !spatial_match_column_takes(g, grid)) return false;
    switch (g.group) {
    case 8: return launch_column<7>(plane, g, grid, ref_begin, ref_count, out, counts, stream);
    case 16: return launch_column<15>(plane, g, grid, ref_begin, ref_count, out, counts, stream);
    case 32: return launch_column<31>(plane, g, grid, ref_begin, ref_count, out, counts, stream);
    default: return false;
    }
}

}  // namespace nss_cuda
