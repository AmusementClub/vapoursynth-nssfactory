// SPDX-License-Identifier: GPL-2.0-only
// Candidate-parallel exact matcher for one-channel spatial searches of blocks
// 2 to 8 (BM3D's default block 8 / step 8 / bm_range 9, and the windows the
// column, square and tensor matchers leave to the lane kernels), also as the
// centre-frame search of the predictive temporal search.
//
// The lane kernels spread one candidate over eight lanes (a block column each)
// and total it by three shuffles, then vote on its key: about 4.8k warp
// instructions a reference at the default shape, bound by load latency and
// the shuffles (Nsight Compute, RTX 5080: 60% issue, IPC 2.6). Here a warp
// takes one reference: its search region is staged in shared memory, and each
// lane computes whole candidates, NY vertically adjacent ones of one column
// at a time (an item), from the region rows that item reads once. A candidate
// keeps B column chains in registers, added row by row from the top as the
// lane kernel's lane j adds column j, and totals them in the lane kernel's
// xor-tree order over the block width rounded up to a power of two (the
// missing columns add exact zeros there and are left out here): the distance
// is the lane kernel's, bit for bit.
// Keys are the lane kernels' (distance bits capped at +inf, then y, then x),
// unique within a window, so the K smallest form the same set whatever the
// order the candidates are seen in: the warp holds them as the lane kernels
// do (lane i the i-th), and each round inserts the keys below the K-th,
// smallest first.
#include "cuda/common/match.hpp"
#include "cuda/runtime/error.hpp"

#include <algorithm>

namespace nss_cuda {
namespace {

constexpr int kMinBlock = 2, kMaxBlock = 8;
constexpr int kWarps = 4;       // references (warps) a thread block
constexpr int kMaxNY = 6;       // candidates of an item (one column, consecutive rows)
constexpr int kMaxSide = 33;    // window sides up to bm_range 16
constexpr unsigned kAll = 0xffffffffu;
constexpr unsigned kInfinity = 0x7f800000u;
constexpr unsigned long long kEmpty = ~0ull;

// Region pitch: at least side + block - 1 and odd, so that the bands of
// items a warp reads at once (rows NY apart) fall on shifted banks.
__host__ __device__ constexpr int region_pitch(int side, int block) {
    const int p = side + block - 1;
    return p | 1;
}

// The lane kernel's xor tree over columns [Lo, Lo + N) of a block of B
// columns (N a power of two); columns at or past B are its idle lanes, whose
// +0 sums leave every partial sum as it is.
template <int B, int Lo, int N>
__device__ __forceinline__ float column_tree(const float (&c)[B]) {
    if constexpr (N == 1) {
        return c[Lo];
    } else if constexpr (Lo + N / 2 >= B) {
        return column_tree<B, Lo, N / 2>(c);
    } else {
        return column_tree<B, Lo, N / 2>(c) + column_tree<B, Lo + N / 2, N / 2>(c);
    }
}

__host__ __device__ constexpr int pow2_at_least(int v) {
    int p = 1;
    while (p < v) p <<= 1;
    return p;
}

__device__ __forceinline__ void order(unsigned long long& a, unsigned long long& b) {
    const unsigned long long lo = a < b ? a : b;
    b = a < b ? b : a;
    a = lo;
}

// The plane searched: `plane`, or frame `frame` of `frames` (the centre of a
// temporal window) when `frames` is set.
template <int B, int NY, bool Table>
__global__ void __launch_bounds__(32 * kWarps) window_match_kernel(const float* plane_arg, const float* const* frames,
                                                                  int frame, MatchGeometry g, RasterGrid grid,
                                                                  int ref_begin, int ref_count, DeviceMatch* out,
                                                                  int* counts) {
    extern __shared__ float regions[];
    const float* const plane = Table ? frames[frame] : plane_arg;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int r = blockIdx.x * kWarps + warp;
    if (r >= ref_count) return;  // whole warps: every lane leaves together
    const int lo = g.lo(), hi = g.hi(), side = lo + hi + 1;
    const int rows = side + B - 1, pitch = region_pitch(side, B);
    float* region = regions + warp * rows * pitch;
    const int cx = grid.x(ref_begin + r), cy = grid.y(ref_begin + r);
    const int x_lo = cx - lo, y_lo = cy - lo;
    const int last_col = g.width - 1, last_row = g.height - 1;
    // The search region, clamped at the plane's edges (only positions
    // outside the plane read the clamped values, and they never enter).
    for (int rx = lane; rx < rows; rx += 32) {
        const float* column = plane + min(max(x_lo + rx, 0), last_col);
        for (int ry = 0; ry < rows; ++ry) {
            region[ry * pitch + rx] = column[static_cast<long long>(min(max(y_lo + ry, 0), last_row)) * g.pitch];
        }
    }
    __syncwarp();
    float ref[B][B];
#pragma unroll
    for (int y = 0; y < B; ++y) {
#pragma unroll
        for (int c = 0; c < B; ++c) ref[y][c] = region[(lo + y) * pitch + lo + c];
    }

    const int max_x = g.width - B, max_y = g.height - B;
    const unsigned self = static_cast<unsigned>(cy) << 16 | static_cast<unsigned>(cx);
    const int wanted = min(g.group, 32) - 1;  // the list's length
    const int bands = (side + NY - 1) / NY, items = side * bands;
    unsigned long long held = kEmpty;
    for (int first = 0; first < items; first += 32) {
        const int item = first + lane;
        const bool live = item < items;
        const int col = live ? item % side : 0, band = live ? item / side : 0;
        const int y0 = band * NY;
        float acc[NY][B];
#pragma unroll
        for (int k = 0; k < NY; ++k) {
#pragma unroll
            for (int c = 0; c < B; ++c) acc[k][c] = 0.f;
        }
        // Region row y0 + i is row i - k of candidate k's block.
#pragma unroll
        for (int i = 0; i < NY + B - 1; ++i) {
            const float* row = region + min(y0 + i, rows - 1) * pitch + col;
            float v[B];
#pragma unroll
            for (int c = 0; c < B; ++c) v[c] = row[c];
#pragma unroll
            for (int k = 0; k < NY; ++k) {
                const int y = i - k;
                if (y < 0 || y >= B) continue;
#pragma unroll
                for (int c = 0; c < B; ++c) {
                    const float d = ref[y][c] - v[c];
                    acc[k][c] = fmaf(d, d, acc[k][c]);
                }
            }
        }
        unsigned long long key[NY];
#pragma unroll
        for (int k = 0; k < NY; ++k) {
            const int x = x_lo + col, y = y_lo + y0 + k;
            const unsigned at = static_cast<unsigned>(y) << 16 | static_cast<unsigned>(x);
            const bool ok = live && y0 + k < side && x >= 0 && x <= max_x && y >= 0 && y <= max_y && at != self;
            key[k] = ok ? static_cast<unsigned long long>(min(__float_as_uint(column_tree<B, 0, pow2_at_least(B)>(acc[k])), kInfinity)) << 32 | at
                        : kEmpty;
        }
        // The lane's keys in ascending order (a transposition network).
#pragma unroll
        for (int pass = 0; pass < NY - 1; ++pass) {
#pragma unroll
            for (int k = 0; k + 1 < NY - pass; ++k) order(key[k], key[k + 1]);
        }
        // Insert the round's keys below the list's last, smallest first: the
        // lane with the smallest head gives it up and moves to its next key.
        unsigned long long bound = __shfl_sync(kAll, held, wanted - 1);
        while (true) {
            const bool enters = key[0] < bound;
            if (!__any_sync(kAll, enters)) break;
#if __CUDA_ARCH__ >= 800
            const unsigned top = __reduce_min_sync(kAll, enters ? static_cast<unsigned>(key[0] >> 32) : ~0u);
            const unsigned low = __reduce_min_sync(
                kAll, enters && static_cast<unsigned>(key[0] >> 32) == top ? static_cast<unsigned>(key[0]) : ~0u);
            const unsigned long long next = static_cast<unsigned long long>(top) << 32 | low;
#else
            const unsigned long long next = __shfl_sync(kAll, key[0], __ffs(__ballot_sync(kAll, enters)) - 1);
#endif
            const unsigned long long pre = __shfl_up_sync(kAll, held, 1);
            if (next < held) held = (lane == 0 || !(next < pre)) ? next : pre;
            if (key[0] == next) {
#pragma unroll
                for (int k = 0; k < NY - 1; ++k) key[k] = key[k + 1];
                key[NY - 1] = kEmpty;
            }
            bound = __shfl_sync(kAll, held, wanted - 1);
        }
    }
    const bool mine = lane < wanted && held != kEmpty;
    const int filled = __popc(__ballot_sync(kAll, mine));
    DeviceMatch* dst = out + static_cast<long long>(r) * g.group;
    if (mine) {
        const unsigned at = static_cast<unsigned>(held), bits = static_cast<unsigned>(held >> 32);
        dst[1 + lane] = DeviceMatch{static_cast<int>(at & 0xffffu), static_cast<int>(at >> 16), 0, __uint_as_float(bits)};
    }
    if (lane == 0) {
        dst[0] = DeviceMatch{cx, cy, 0, 0.f};
        counts[r] = 1 + filled;
    }
}

// Rows an item takes for a window side and block: the fewest instructions by
// a rough count a lane makes (rounds of 32 items, each NY candidates of 2 B^2
// subtractions and fused multiply-adds, its region rows' B loads each and
// some 40 for its keys).
int item_rows(int side, int block) {
    int best = 2;
    long long best_cost = -1;
    for (int ny = 2; ny <= kMaxNY; ++ny) {
        const long long rounds = (static_cast<long long>(side) * ((side + ny - 1) / ny) + 31) / 32;
        const long long cost = rounds * (ny * 2 * block * block + (ny + block - 1) * block + 40);
        if (best_cost < 0 || cost < best_cost) best = ny, best_cost = cost;
    }
    return best;
}

std::size_t region_bytes(const MatchGeometry& g) {
    const int side = g.lo() + g.hi() + 1;
    return static_cast<std::size_t>(side + g.block - 1) * region_pitch(side, g.block) * sizeof(float);
}

template <int B>
void launch_block(const float* plane, const float* const* frames, int frame, const MatchGeometry& g,
                  const RasterGrid& grid, int ref_begin, int ref_count, DeviceMatch* out, int* counts,
                  cudaStream_t stream) {
    const int blocks = (ref_count + kWarps - 1) / kWarps;
    const std::size_t shared = kWarps * region_bytes(g);
    switch (item_rows(g.lo() + g.hi() + 1, B)) {
#define NSS_WINDOW_NY(N)                                                                                     \
    case N:                                                                                                  \
        if (frames) {                                                                                        \
            window_match_kernel<B, N, true><<<blocks, 32 * kWarps, shared, stream>>>(                        \
                plane, frames, frame, g, grid, ref_begin, ref_count, out, counts);                           \
        } else {                                                                                             \
            window_match_kernel<B, N, false><<<blocks, 32 * kWarps, shared, stream>>>(                       \
                plane, frames, frame, g, grid, ref_begin, ref_count, out, counts);                           \
        }                                                                                                    \
        break;
        NSS_WINDOW_NY(2)
        NSS_WINDOW_NY(3)
        NSS_WINDOW_NY(4)
        NSS_WINDOW_NY(5)
        NSS_WINDOW_NY(6)
#undef NSS_WINDOW_NY
    }
    NSS_CUDA_CHECK_LAUNCH();
}

bool run_window(const float* plane, const float* const* frames, int frame, const MatchGeometry& g,
                const RasterGrid& grid, int ref_begin, int ref_count, DeviceMatch* out, int* counts,
                cudaStream_t stream) {
    if (!spatial_match_window_takes(g, grid)) return false;
    if (ref_count <= 0) return true;
    switch (g.block) {
#define NSS_WINDOW_B(B)                                                                                     \
    case B: launch_block<B>(plane, frames, frame, g, grid, ref_begin, ref_count, out, counts, stream); break;
        NSS_WINDOW_B(2)
        NSS_WINDOW_B(3)
        NSS_WINDOW_B(4)
        NSS_WINDOW_B(5)
        NSS_WINDOW_B(6)
        NSS_WINDOW_B(7)
        NSS_WINDOW_B(8)
#undef NSS_WINDOW_B
    default: return false;
    }
    return true;
}

// The smallest window side per block where the kernel is at least as fast
// as the lane kernels, with a margin (1080p gray, RTX 5080, groups 8 and 16
// at steps block and block / 2: below these sides it is 0.37-1.0x; at them
// 1.02-1.75x, and 1.3-2.8x at sides 19 to 25).
int window_min_side(int block) {
    constexpr int kSide[kMaxBlock + 1] = {0, 0, 13, 9, 9, 9, 9, 11, 15};
    return kSide[block];
}

}  // namespace

bool spatial_match_window_takes(const MatchGeometry& g, const RasterGrid& grid) {
    (void)grid;
    const int side = g.lo() + g.hi() + 1;
    return g.channels == 1 && g.block >= kMinBlock && g.block <= kMaxBlock && g.group >= 2 && g.group <= 32 &&
           side >= window_min_side(g.block) && side <= kMaxSide && g.width >= g.block && g.height >= g.block &&
           g.width <= 0xffff && g.height <= 0xffff;
}

bool predictive_match_window_takes(const MatchGeometry& g, const RasterGrid& grid) {
    // Seeding pays from a larger window than the spatial search (the extra
    // launch): BM3D / WNNM radius 1 at block 8, side 15 measured 0.99-1.01x,
    // side 19 1.17x (kernels) with the lane kernels' bits either way.
    return spatial_match_window_takes(g, grid) && g.lo() + g.hi() + 1 >= window_min_side(g.block) + 4;
}

bool spatial_match_window(const float* plane, const MatchGeometry& g, const RasterGrid& grid, int ref_begin,
                          int ref_count, DeviceMatch* out, int* counts, cudaStream_t stream) {
    return run_window(plane, nullptr, 0, g, grid, ref_begin, ref_count, out, counts, stream);
}

bool spatial_match_window_frame(const float* const* frames, int frame, const MatchGeometry& g, const RasterGrid& grid,
                                int ref_begin, int ref_count, DeviceMatch* out, int* counts, cudaStream_t stream) {
    return run_window(nullptr, frames, frame, g, grid, ref_begin, ref_count, out, counts, stream);
}

}  // namespace nss_cuda
