// SPDX-License-Identifier: GPL-2.0-only
// Candidate-parallel exact matcher for one-channel spatial searches, also as
// the centre-frame search of the predictive temporal search.
//
// The lane kernels spread one candidate over eight lanes (a block column each)
// and total it by three shuffles, then vote on its key: about 4.8k warp
// instructions a reference at the default shape, bound by load latency and
// the shuffles (Nsight Compute, RTX 5080: 60% issue, IPC 2.6). Here a warp
// takes one reference. Its search window is staged in shared memory a tile of
// at most kTile x kTile candidates at a time, and each lane computes whole
// candidates, NY vertically adjacent ones of one column at a time (an item),
// from the region rows that item reads once.
//
// Every mode reproduces the arithmetic of the kernel it replaces, so the
// distances are the same bits:
// - Tree (blocks 2 to 8, groups up to 33: the lane kernels). A candidate
//   keeps B column chains, added row by row from the top as the lane kernel's
//   lane j adds column j, and totals them in the lane kernel's xor-tree order
//   over the block width rounded up to a power of two (the missing columns add
//   exact zeros there and are left out here).
// - Chain (blocks 2 to 8, groups 34 to 64: the select and block top-K
//   kernels). One row-major chain a candidate (block_ssd's order).
// - Split (blocks 12 and 16, groups up to 33: the lane kernels). Two lanes a
//   candidate, eight block columns each in registers. With eight lanes (groups
//   up to 9) lane j of the lane kernel chains its columns j and j + 8 row by
//   row: half h takes the lane kernel's lanes 4h..4h+3. With 16 (more) each
//   column has its chain and a 16-wide tree: half h takes columns 8h..8h+7.
//   Either way a half's partial is the lower part of the lane kernel's tree,
//   and the two add by one shuffle as its last step does.
// Keys are those kernels' (distance bits capped at +inf, then y, then x; the
// non-finite sort as +inf, after the finite), unique within a window, so the
// K smallest form the same set whatever the order the candidates are seen in:
// the warp holds them as the lane kernels do (rank r on lane r % 32, in slot
// r / 32), and each round inserts the keys below the K-th, smallest first.
#include "cuda/common/match.hpp"
#include "cuda/runtime/error.hpp"

#include <algorithm>

namespace nss_cuda {
namespace {

constexpr int kWarps = 4;     // references (warps) a thread block
constexpr int kMaxNY = 6;     // candidates of an item (one column, consecutive rows)
constexpr int kTile = 33;     // candidates a tile side
constexpr int kMaxSide = 129; // window sides up to bm_range 64
constexpr unsigned kAll = 0xffffffffu;
constexpr unsigned kInfinity = 0x7f800000u;
constexpr unsigned long long kEmpty = ~0ull;

enum class Mode { Tree, Chain, Split };

// Region pitch: at least the region's width and odd, so that the bands of
// items a warp reads at once (rows NY apart) fall on shifted banks.
__host__ __device__ constexpr int region_pitch(int width) { return width | 1; }

// The lane kernel's xor tree over columns [Lo, Lo + N) of an array of B
// column sums (N a power of two); columns at or past B are its idle lanes,
// whose +0 sums leave every partial sum as it is.
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

// The best keys of the warp, rank r on lane r % 32 in slot r / 32.
template <int Slots>
struct HeldList {
    unsigned long long h[Slots];
    __device__ __forceinline__ void init() {
#pragma unroll
        for (int s = 0; s < Slots; ++s) h[s] = kEmpty;
    }
    // Every lane: the key of rank `rank`.
    __device__ __forceinline__ unsigned long long at(int rank) const {
        unsigned long long v = h[0];
        if constexpr (Slots > 1) v = rank >= 32 ? h[1] : h[0];
        return __shfl_sync(kAll, v, rank & 31);
    }
    // Every lane, the same `next`: the entries from its rank on move up one.
    __device__ __forceinline__ void insert(unsigned long long next, int lane) {
        const unsigned long long pre0 = __shfl_up_sync(kAll, h[0], 1);
        if constexpr (Slots > 1) {
            const unsigned long long top0 = __shfl_sync(kAll, h[0], 31);
            const unsigned long long up1 = __shfl_up_sync(kAll, h[1], 1);
            const unsigned long long pre1 = lane == 0 ? top0 : up1;
            if (next < h[1]) h[1] = !(next < pre1) ? next : pre1;
        }
        if (next < h[0]) h[0] = (lane == 0 || !(next < pre0)) ? next : pre0;
    }
};

// A lane's NY keys of a round into the list: sorted, then the keys below the
// list's K-th, the smallest of all lanes first.
template <int NY, int Slots>
__device__ __forceinline__ void insert_round(unsigned long long (&key)[NY], HeldList<Slots>& held, int wanted,
                                             int lane) {
#pragma unroll
    for (int pass = 0; pass < NY - 1; ++pass) {
#pragma unroll
        for (int k = 0; k + 1 < NY - pass; ++k) order(key[k], key[k + 1]);
    }
    unsigned long long bound = held.at(wanted - 1);
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
        held.insert(next, lane);
        if (key[0] == next) {
#pragma unroll
            for (int k = 0; k < NY - 1; ++k) key[k] = key[k + 1];
            key[NY - 1] = kEmpty;
        }
        bound = held.at(wanted - 1);
    }
}

template <int Slots>
__device__ __forceinline__ void write_list(const HeldList<Slots>& held, int wanted, int lane, int cx, int cy, int t,
                                           DeviceMatch* dst, int* count) {
    int filled = 0;
#pragma unroll
    for (int s = 0; s < Slots; ++s) {
        const int rank = 32 * s + lane;
        const bool mine = rank < wanted && held.h[s] != kEmpty;
        filled += __popc(__ballot_sync(kAll, mine));
        if (mine) {
            const unsigned at = static_cast<unsigned>(held.h[s]), bits = static_cast<unsigned>(held.h[s] >> 32);
            dst[1 + rank] = DeviceMatch{static_cast<int>(at & 0xffffu), static_cast<int>(at >> 16), t,
                                        __uint_as_float(bits)};
        }
    }
    if (lane == 0) {
        dst[0] = DeviceMatch{cx, cy, t, 0.f};
        *count = 1 + filled;
    }
}

// The window's tiles: `cols` x `rows` tiles of at most tile_w x tile_h
// candidates.
struct Tiling {
    int tile_w, tile_h, cols, rows;
};

// Stages the region of the tile at (ox, oy) of the window: (ch + B - 1) rows
// of (cw + B - 1) samples, clamped at the plane's edges (only positions
// outside the plane read the clamped values, and they never enter).
__device__ __forceinline__ void stage(float* region, int pitch, const float* plane, const MatchGeometry& g, int x0,
                                      int y0, int width, int height, int lane) {
    const int last_col = g.width - 1, last_row = g.height - 1;
    __syncwarp();  // the previous tile's reads are done
    for (int rx = lane; rx < width; rx += 32) {
        const float* column = plane + min(max(x0 + rx, 0), last_col);
        for (int ry = 0; ry < height; ++ry) {
            region[ry * pitch + rx] = column[static_cast<long long>(min(max(y0 + ry, 0), last_row)) * g.pitch];
        }
    }
    __syncwarp();
}

// Tree and Chain: one lane a candidate, blocks 2 to 8. The plane searched is
// `plane`, or frame `frame` of `frames` (the centre of a temporal window).
template <int B, int NY, bool Table, Mode M>
__global__ void __launch_bounds__(32 * kWarps) window_match_kernel(const float* plane_arg, const float* const* frames,
                                                                  int frame, MatchGeometry g, RasterGrid grid,
                                                                  Tiling tiling, int ref_begin, int ref_count,
                                                                  DeviceMatch* out, int* counts) {
    static_assert(M != Mode::Split);
    constexpr int kSlots = M == Mode::Chain ? 2 : 1;
    extern __shared__ float regions[];
    const float* const plane = Table ? frames[frame] : plane_arg;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int r = blockIdx.x * kWarps + warp;
    if (r >= ref_count) return;  // whole warps: every lane leaves together
    const int lo = g.lo(), side = lo + g.hi() + 1;
    const int pitch = region_pitch(tiling.tile_w + B - 1);
    float* region = regions + warp * (tiling.tile_h + B - 1) * pitch;
    const int cx = grid.x(ref_begin + r), cy = grid.y(ref_begin + r);
    const int x_lo = cx - lo, y_lo = cy - lo;
    float ref[B][B];
#pragma unroll
    for (int y = 0; y < B; ++y) {
#pragma unroll
        for (int c = 0; c < B; ++c) ref[y][c] = plane[static_cast<long long>(cy + y) * g.pitch + cx + c];
    }
    const int max_x = g.width - B, max_y = g.height - B;
    const unsigned self = static_cast<unsigned>(cy) << 16 | static_cast<unsigned>(cx);
    const int wanted = min(g.group, 32 * kSlots + 1) - 1;  // the list's length
    HeldList<kSlots> held;
    held.init();
    for (int ty = 0; ty < tiling.rows; ++ty) {
        const int oy = ty * tiling.tile_h, ch = min(tiling.tile_h, side - oy);
        for (int tx = 0; tx < tiling.cols; ++tx) {
            const int ox = tx * tiling.tile_w, cw = min(tiling.tile_w, side - ox);
            const int rows = ch + B - 1;
            stage(region, pitch, plane, g, x_lo + ox, y_lo + oy, cw + B - 1, rows, lane);
            const int bands = (ch + NY - 1) / NY, items = cw * bands;
            for (int first = 0; first < items; first += 32) {
                const int item = first + lane;
                const bool live = item < items;
                const int col = live ? item % cw : 0, band = live ? item / cw : 0;
                const int y0 = band * NY;
                float acc[NY][M == Mode::Tree ? B : 1];
#pragma unroll
                for (int k = 0; k < NY; ++k) {
#pragma unroll
                    for (int c = 0; c < (M == Mode::Tree ? B : 1); ++c) acc[k][c] = 0.f;
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
                            if constexpr (M == Mode::Tree) {
                                acc[k][c] = fmaf(d, d, acc[k][c]);
                            } else {
                                acc[k][0] = fmaf(d, d, acc[k][0]);
                            }
                        }
                    }
                }
                unsigned long long key[NY];
#pragma unroll
                for (int k = 0; k < NY; ++k) {
                    const int x = x_lo + ox + col, y = y_lo + oy + y0 + k;
                    const unsigned at = static_cast<unsigned>(y) << 16 | static_cast<unsigned>(x);
                    const bool ok = live && y0 + k < ch && x >= 0 && x <= max_x && y >= 0 && y <= max_y && at != self;
                    float sum;
                    if constexpr (M == Mode::Tree) {
                        sum = column_tree<B, 0, pow2_at_least(B)>(acc[k]);
                    } else {
                        sum = acc[k][0];
                    }
                    key[k] = ok ? static_cast<unsigned long long>(min(__float_as_uint(sum), kInfinity)) << 32 | at
                                : kEmpty;
                }
                insert_round<NY, kSlots>(key, held, wanted, lane);
            }
        }
    }
    write_list<kSlots>(held, wanted, lane, cx, cy, 0, out + static_cast<long long>(r) * g.group, counts + r);
}

// Split: two lanes a candidate, blocks 12 and 16 (see the file comment). Pair:
// the lane kernels' eight-lane layout (groups up to 9).
template <int B, int NY, bool Table, bool Pair>
__global__ void __launch_bounds__(32 * kWarps) window_split_kernel(const float* plane_arg, const float* const* frames,
                                                                  int frame, MatchGeometry g, RasterGrid grid,
                                                                  Tiling tiling, int ref_begin, int ref_count,
                                                                  DeviceMatch* out, int* counts) {
    static_assert(B == 12 || B == 16);
    extern __shared__ float regions[];
    const float* const plane = Table ? frames[frame] : plane_arg;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, half = lane & 1;
    const int r = blockIdx.x * kWarps + warp;
    if (r >= ref_count) return;
    const int lo = g.lo(), side = lo + g.hi() + 1;
    const int pitch = region_pitch(tiling.tile_w + B - 1);
    float* region = regions + warp * (tiling.tile_h + B - 1) * pitch;
    const int cx = grid.x(ref_begin + r), cy = grid.y(ref_begin + r);
    const int x_lo = cx - lo, y_lo = cy - lo;
    // The half's eight block columns q: Pair, the lane kernel's lanes
    // j = 4 half + (q & 3), column j for q < 4 and j + 8 above; else column
    // 8 half + q. Columns at or past B are absent (zero terms).
    int column[8];
    bool has[8];
#pragma unroll
    for (int q = 0; q < 8; ++q) {
        column[q] = Pair ? 4 * half + (q & 3) + (q < 4 ? 0 : 8) : 8 * half + q;
        has[q] = column[q] < B;
        column[q] = has[q] ? column[q] : 0;
    }
    float ref[B][8];
#pragma unroll
    for (int y = 0; y < B; ++y) {
#pragma unroll
        for (int q = 0; q < 8; ++q) ref[y][q] = plane[static_cast<long long>(cy + y) * g.pitch + cx + column[q]];
    }
    const int max_x = g.width - B, max_y = g.height - B;
    const unsigned self = static_cast<unsigned>(cy) << 16 | static_cast<unsigned>(cx);
    const int wanted = min(g.group, 33) - 1;
    constexpr int kChains = Pair ? 4 : 8;
    HeldList<1> held;
    held.init();
    for (int ty = 0; ty < tiling.rows; ++ty) {
        const int oy = ty * tiling.tile_h, ch = min(tiling.tile_h, side - oy);
        for (int tx = 0; tx < tiling.cols; ++tx) {
            const int ox = tx * tiling.tile_w, cw = min(tiling.tile_w, side - ox);
            const int rows = ch + B - 1;
            stage(region, pitch, plane, g, x_lo + ox, y_lo + oy, cw + B - 1, rows, lane);
            const int bands = (ch + NY - 1) / NY, items = cw * bands;
            for (int first = 0; first < items; first += 16) {
                const int item = first + lane / 2;
                const bool live = item < items;
                const int col = live ? item % cw : 0, band = live ? item / cw : 0;
                const int y0 = band * NY;
                float acc[NY][kChains];
#pragma unroll
                for (int k = 0; k < NY; ++k) {
#pragma unroll
                    for (int c = 0; c < kChains; ++c) acc[k][c] = 0.f;
                }
#pragma unroll
                for (int i = 0; i < NY + B - 1; ++i) {
                    const float* row = region + min(y0 + i, rows - 1) * pitch + col;
                    float v[8];
#pragma unroll
                    for (int q = 0; q < 8; ++q) v[q] = row[column[q]];
#pragma unroll
                    for (int k = 0; k < NY; ++k) {
                        const int y = i - k;
                        if (y < 0 || y >= B) continue;
                        // Pair: chain c takes column c, then c + 8 (row by row).
#pragma unroll
                        for (int q = 0; q < 8; ++q) {
                            if (!has[q]) continue;
                            const float d = ref[y][q] - v[q];
                            float& a = acc[k][Pair ? (q & 3) : q];
                            a = fmaf(d, d, a);
                        }
                    }
                }
                unsigned long long key[NY];
#pragma unroll
                for (int k = 0; k < NY; ++k) {
                    float part;
                    if constexpr (Pair) {
                        part = (acc[k][0] + acc[k][1]) + (acc[k][2] + acc[k][3]);
                    } else {
                        part = ((acc[k][0] + acc[k][1]) + (acc[k][2] + acc[k][3])) +
                               ((acc[k][4] + acc[k][5]) + (acc[k][6] + acc[k][7]));
                    }
                    const float sum = part + __shfl_xor_sync(kAll, part, 1);
                    const int x = x_lo + ox + col, y = y_lo + oy + y0 + k;
                    const unsigned at = static_cast<unsigned>(y) << 16 | static_cast<unsigned>(x);
                    const bool ok = half == 0 && live && y0 + k < ch && x >= 0 && x <= max_x && y >= 0 && y <= max_y &&
                                    at != self;
                    key[k] = ok ? static_cast<unsigned long long>(min(__float_as_uint(sum), kInfinity)) << 32 | at
                                : kEmpty;
                }
                insert_round<NY, 1>(key, held, wanted, lane);
            }
        }
    }
    write_list<1>(held, wanted, lane, cx, cy, 0, out + static_cast<long long>(r) * g.group, counts + r);
}

// The smallest window side where the kernel is at least as fast as the
// kernel it replaces, with a margin (1080p gray, RTX 5080,
// tests/cuda/bench_cuda_match):
// - Tree, against the lane kernels (groups 8 and 16 at steps block and
//   block / 2): 0.37-1.0x below these sides, 1.02-1.75x at them, 1.3-2.8x at
//   sides 19 to 25 and 1.6-2.6x up to 129.
// - Chain, against the select and block top-K kernels: 1.15-7.5x at every
//   side (5x at side 5, 7x at 129).
// - Split, against the lane kernels: block 12 0.40-0.94x below side 19,
//   1.21-1.64x from it; block 16 with 16 lanes 0.38-0.95x below 25,
//   1.05-1.39x from it; with eight (groups up to 9) 0.39-0.91x up to side 33,
//   1.07x at 49.
int min_side(Mode m, int block, int group) {
    constexpr int kOff = 1 << 30;
    switch (m) {
    case Mode::Tree: {
        constexpr int kSide[9] = {kOff, kOff, 13, 9, 9, 9, 9, 11, 15};
        return block >= 0 && block <= 8 ? kSide[block] : kOff;
    }
    case Mode::Chain: return 1;
    case Mode::Split: return block == 12 ? 19 : group <= 9 ? 49 : 25;
    }
    return kOff;
}

bool mode_of(const MatchGeometry& g, Mode& m) {
    if (g.channels != 1 || g.group < 2 || g.group > 64) return false;
    if (g.block >= 2 && g.block <= 8) {
        m = g.group <= 33 ? Mode::Tree : Mode::Chain;
        return true;
    }
    if ((g.block == 12 || g.block == 16) && g.group <= 33) {
        m = Mode::Split;
        return true;
    }
    return false;
}

Tiling tiling_of(int side) {
    const int cols = (side + kTile - 1) / kTile, tile = (side + cols - 1) / cols;
    return Tiling{tile, tile, cols, cols};
}

// Rows an item takes for a tile side and block: the fewest instructions by a
// rough count a lane makes (rounds of `per_round` items, each NY candidates of
// 2 B^2 / lanes subtractions and fused multiply-adds, its region rows' loads
// and some 40 for its keys).
int item_rows(int tile, int block, int lanes_per, int max_ny) {
    const int per_round = 32 / lanes_per, cols = block / lanes_per;
    int best = 2;
    long long best_cost = -1;
    for (int ny = 2; ny <= max_ny; ++ny) {
        const long long rounds = (static_cast<long long>(tile) * ((tile + ny - 1) / ny) + per_round - 1) / per_round;
        const long long cost = rounds * (ny * 2 * block * cols + (ny + block - 1) * cols + 40);
        if (best_cost < 0 || cost < best_cost) best = ny, best_cost = cost;
    }
    return best;
}

std::size_t region_bytes(const MatchGeometry& g, const Tiling& t) {
    return static_cast<std::size_t>(t.tile_h + g.block - 1) * region_pitch(t.tile_w + g.block - 1) * sizeof(float);
}

template <int B, Mode M, bool Table>
void launch_block(const float* plane, const float* const* frames, int frame, const MatchGeometry& g,
                  const RasterGrid& grid, int ref_begin, int ref_count, DeviceMatch* out, int* counts,
                  cudaStream_t stream) {
    const int blocks = (ref_count + kWarps - 1) / kWarps;
    const Tiling t = tiling_of(g.lo() + g.hi() + 1);
    const std::size_t shared = kWarps * region_bytes(g, t);
    if constexpr (M == Mode::Split) {
        const bool pair = g.group <= 9;  // lane_group: eight lanes up to 8 matches
        switch (item_rows(t.tile_h, B, 2, 4)) {
#define NSS_WINDOW_SPLIT(N)                                                                                         \
    case N:                                                                                                         \
        if (pair) {                                                                                                 \
            window_split_kernel<B, N, Table, true><<<blocks, 32 * kWarps, shared, stream>>>(                        \
                plane, frames, frame, g, grid, t, ref_begin, ref_count, out, counts);                               \
        } else {                                                                                                    \
            window_split_kernel<B, N, Table, false><<<blocks, 32 * kWarps, shared, stream>>>(                       \
                plane, frames, frame, g, grid, t, ref_begin, ref_count, out, counts);                               \
        }                                                                                                           \
        break;
            NSS_WINDOW_SPLIT(2)
            NSS_WINDOW_SPLIT(3)
            NSS_WINDOW_SPLIT(4)
#undef NSS_WINDOW_SPLIT
        }
    } else {
        switch (item_rows(t.tile_h, B, 1, kMaxNY)) {
#define NSS_WINDOW_NY(N)                                                                                            \
    case N:                                                                                                         \
        window_match_kernel<B, N, Table, M><<<blocks, 32 * kWarps, shared, stream>>>(plane, frames, frame, g, grid, \
                                                                                    t, ref_begin, ref_count, out,    \
                                                                                    counts);                         \
        break;
            NSS_WINDOW_NY(2)
            NSS_WINDOW_NY(3)
            NSS_WINDOW_NY(4)
            NSS_WINDOW_NY(5)
            NSS_WINDOW_NY(6)
#undef NSS_WINDOW_NY
        }
    }
    NSS_CUDA_CHECK_LAUNCH();
}

template <bool Table>
bool run_window(const float* plane, const float* const* frames, int frame, const MatchGeometry& g,
                const RasterGrid& grid, int ref_begin, int ref_count, DeviceMatch* out, int* counts,
                cudaStream_t stream) {
    Mode m{};
    if (!spatial_match_window_takes(g, grid) || !mode_of(g, m)) return false;
    if (ref_count <= 0) return true;
    const auto args = [&](auto launch) {
        launch(plane, frames, frame, g, grid, ref_begin, ref_count, out, counts, stream);
    };
    if (m == Mode::Split) {
        if (g.block == 12) args(launch_block<12, Mode::Split, Table>);
        else args(launch_block<16, Mode::Split, Table>);
        return true;
    }
    if constexpr (!Table) {
        if (m == Mode::Chain) {
            switch (g.block) {
#define NSS_WINDOW_B(B) \
    case B: args(launch_block<B, Mode::Chain, false>); break;
                NSS_WINDOW_B(2) NSS_WINDOW_B(3) NSS_WINDOW_B(4) NSS_WINDOW_B(5) NSS_WINDOW_B(6) NSS_WINDOW_B(7)
                NSS_WINDOW_B(8)
#undef NSS_WINDOW_B
            }
            return true;
        }
    } else {
        if (m == Mode::Chain) return false;  // the predictive lane kernels take groups up to 33
    }
    switch (g.block) {
#define NSS_WINDOW_B(B) \
    case B: args(launch_block<B, Mode::Tree, Table>); break;
        NSS_WINDOW_B(2) NSS_WINDOW_B(3) NSS_WINDOW_B(4) NSS_WINDOW_B(5) NSS_WINDOW_B(6) NSS_WINDOW_B(7) NSS_WINDOW_B(8)
#undef NSS_WINDOW_B
    }
    return true;
}

}  // namespace

bool spatial_match_window_takes(const MatchGeometry& g, const RasterGrid& grid) {
    (void)grid;
    Mode m{};
    const int side = g.lo() + g.hi() + 1;
    return mode_of(g, m) && side >= min_side(m, g.block, g.group) && side <= kMaxSide && g.width >= g.block &&
           g.height >= g.block && g.width <= 0xffff && g.height <= 0xffff;
}

bool predictive_match_window_takes(const MatchGeometry& g, const RasterGrid& grid) {
    // Seeding pays from a larger window than the spatial search (the extra
    // launch): BM3D / WNNM radius 1 at block 8, side 15 measured 0.99-1.01x,
    // side 19 1.17x (kernels) with the lane kernels' bits either way.
    Mode m{};
    // Tree only: the split layout follows the predictive lane kernels' lane
    // count, which ps_num can raise.
    return spatial_match_window_takes(g, grid) && mode_of(g, m) && m == Mode::Tree &&
           g.lo() + g.hi() + 1 >= min_side(m, g.block, g.group) + 4;
}

bool spatial_match_window_over_tensor(const MatchGeometry& g, const RasterGrid& grid) {
    // Where both take the search, the window kernel up to a side of 49:
    // block 8, groups 8 / 16 at side 41 1.43 / 1.51x, 49 1.27 / 1.34x, 65
    // 0.88 / 0.91x, 129 0.58 / 0.82x the tensor-core matcher.
    return spatial_match_window_takes(g, grid) && g.lo() + g.hi() + 1 <= 49;
}

bool spatial_match_window(const float* plane, const MatchGeometry& g, const RasterGrid& grid, int ref_begin,
                          int ref_count, DeviceMatch* out, int* counts, cudaStream_t stream) {
    return run_window<false>(plane, nullptr, 0, g, grid, ref_begin, ref_count, out, counts, stream);
}

bool spatial_match_window_frame(const float* const* frames, int frame, const MatchGeometry& g, const RasterGrid& grid,
                                int ref_begin, int ref_count, DeviceMatch* out, int* counts, cudaStream_t stream) {
    return run_window<true>(nullptr, frames, frame, g, grid, ref_begin, ref_count, out, counts, stream);
}

}  // namespace nss_cuda
