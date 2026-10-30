// SPDX-License-Identifier: GPL-2.0-only
// Square matcher for blocks of R times the step (Davy & Ehret, "GPU
// acceleration of NL-means, BM3D and VBM3D", JRTIP 18 (2021), section 4.1.4).
//
// At step Q an RQ x RQ block is R^2 Q x Q tiles, and each tile is the top-left
// tile of a reference of the grid: its distance for a displacement is shared
// by the R^2 references that contain it. A block of 16 x 16 threads takes
// 16 x 16 lattice positions; every thread computes the distance of its own
// top-left tile for each displacement (its reference tile in registers, the
// compared tile a ring of Q rows in registers, one new row a displacement),
// and the references of the block's top-left corner total their R^2 tiles from
// shared memory. There are 16 - R + 1 references per side. A batch of fewer
// grid rows takes blocks of 16 x (rows + R - 1) threads so that it computes
// only its own rows. A reference keeps its best group - 1 keys sorted in
// registers (the paper's Algorithm 2: insert only when a key beats the last).
//
// Keys and order are the lane kernels' (distance bits capped at +inf, then
// y, then x; the reference first), but a distance is the sum of R^2 tile sums
// (Q-row column chains, then a tree) instead of the lane kernels' block-row
// chains: the last bits differ, so near ties may resolve differently.
//
// Grids whose last column or row is clamped (max_x or max_y not a multiple of
// Q) take extra passes on a lattice shifted to put the clamped references on
// it; a reference is on exactly one pass's lattice.
#include "cuda/common/match.hpp"
#include "cuda/runtime/error.hpp"

#include <algorithm>

namespace nss_cuda {
namespace {

constexpr int kSide = 16;  // threads per block side
constexpr unsigned kInfinity = 0x7f800000u;
constexpr unsigned long long kEmpty = ~0ull;

template <int K>
__device__ __forceinline__ void insert_sorted(unsigned long long (&t)[K], unsigned long long key) {
#pragma unroll
    for (int i = K - 1; i > 0; --i) t[i] = key < t[i - 1] ? t[i - 1] : (key < t[i] ? key : t[i]);
    t[0] = key < t[0] ? key : t[0];
}

// A lattice: position (j, i) of block (bx, by) is the tile at
// (ox + Q ((16 - R + 1) bx + j), oy + Q (rows by + i)), rows the launch's
// references per block column.
struct Pass {
    int ox, oy;
    int blocks_x;     // blocks per lattice row band
    int first_block;  // first block of the launch that runs this pass
};
// The passes of one batch in one launch: the main lattice and up to three
// shifted ones for clamped edges (separate launches of a few blocks each cost
// a whole block's time apiece on small frames).
struct Passes {
    Pass pass[4];
    int count;
};

// The searched plane: `plane`, or frame `frame` of the device pointer table
// `frames` (a temporal window's centre: the table is filled on the stream).
struct Source {
    const float* plane;
    const float* const* frames;
    int frame;
};

// Grid index of the reference at x (y), or -1 when no reference is there.
template <int Q>
__device__ __forceinline__ int grid_index(int v, int max_v, int count) {
    if (v < 0 || v > max_v) return -1;
    if (v == max_v) return count - 1;
    return v % Q == 0 ? v / Q : -1;
}

// A tile's distance from its column sums: a pairwise tree.
template <int N>
__device__ __forceinline__ float tree_sum(const float* c) {
    if constexpr (N == 1) return c[0];
    else return tree_sum<(N + 1) / 2>(c) + tree_sum<N / 2>(c + (N + 1) / 2);
}

template <int R, int Q, int K>
__global__ void __launch_bounds__(kSide * kSide) square_match_kernel(Source source, MatchGeometry g,
                                                                     RasterGrid grid, int ref_begin, int ref_count,
                                                                     int rows, Passes passes, DeviceMatch* out,
                                                                     int* counts) {
    constexpr int refs = kSide - R + 1;
    const float* const plane = source.frames ? source.frames[source.frame] : source.plane;
    __shared__ float part[2][kSide][kSide + 1];
    const int tid = threadIdx.x, tx = tid % kSide, ty = tid / kSide;
    int which = 0;
    for (int i = 1; i < passes.count; ++i) {
        if (static_cast<int>(blockIdx.x) >= passes.pass[i].first_block) which = i;
    }
    const Pass& pass = passes.pass[which];
    const int local = static_cast<int>(blockIdx.x) - pass.first_block;
    const int bx = local % pass.blocks_x;
    const int by = local / pass.blocks_x;
    const int x0 = pass.ox + Q * (refs * bx + tx);
    const int y0 = pass.oy + Q * (rows * by + ty);
    const int col = grid_index<Q>(x0, grid.max_x, grid.nx), row = grid_index<Q>(y0, grid.max_y, grid.ny);
    const int ref = row * grid.nx + col;
    const bool live = tx < refs && ty < rows && col >= 0 && row >= 0 && ref >= ref_begin &&
                      ref < ref_begin + ref_count;
    const int lo = g.lo(), hi = g.hi();
    const int max_x = g.width - R * Q, max_y = g.height - R * Q, last_row = g.height - 1, last_col = g.width - 1;
    const long long pitch = g.pitch;
    int cols[Q];
#pragma unroll
    for (int c = 0; c < Q; ++c) cols[c] = min(max(x0 + c, 0), last_col);
    float rq[Q][Q];
#pragma unroll
    for (int r = 0; r < Q; ++r) {
        const long long at = min(max(y0 + r, 0), last_row) * pitch;
#pragma unroll
        for (int c = 0; c < Q; ++c) rq[r][c] = plane[at + cols[c]];
    }
    unsigned long long t[K];
#pragma unroll
    for (int i = 0; i < K; ++i) t[i] = kEmpty;
    int buf = 0;
    for (int dx = -lo; dx <= hi; ++dx) {
        int xs[Q];
#pragma unroll
        for (int c = 0; c < Q; ++c) xs[c] = min(max(x0 + c + dx, 0), last_col);
        // v[slot][column]: before phase s, row y0 + dy + k is in slot (s + 1 + k) % Q for k < Q - 1.
        float v[Q][Q];
#pragma unroll
        for (int k = 0; k < Q - 1; ++k) {
            const long long at = min(max(y0 - lo + k, 0), last_row) * pitch;
#pragma unroll
            for (int c = 0; c < Q; ++c) v[k + 1][c] = plane[at + xs[c]];
        }
        const int xc = x0 + dx;
        const bool col_ok = live && xc >= 0 && xc <= max_x;
        int dy = -lo;
        while (dy <= hi) {
#pragma unroll
            for (int s = 0; s < Q; ++s) {
                if (dy > hi) break;  // uniform over the block
                const long long at = min(max(y0 + dy + Q - 1, 0), last_row) * pitch;
#pragma unroll
                for (int c = 0; c < Q; ++c) v[s][c] = plane[at + xs[c]];
                float cs[Q];
#pragma unroll
                for (int c = 0; c < Q; ++c) {
                    float sum = 0.f;
#pragma unroll
                    for (int r = 0; r < Q; ++r) {
                        const float d = rq[r][c] - v[(s + 1 + r) % Q][c];
                        sum = fmaf(d, d, sum);
                    }
                    cs[c] = sum;
                }
                part[buf][ty][tx] = tree_sum<Q>(cs);
                __syncthreads();
                const int yc = y0 + dy;
                if (col_ok && yc >= 0 && yc <= max_y && (dx | dy) != 0) {
                    float tiles[R * R];
#pragma unroll
                    for (int r = 0; r < R; ++r) {
#pragma unroll
                        for (int c = 0; c < R; ++c) tiles[r * R + c] = part[buf][ty + r][tx + c];
                    }
                    const float sum = tree_sum<R * R>(tiles);
                    const unsigned long long key =
                        static_cast<unsigned long long>(min(__float_as_uint(sum), kInfinity)) << 32 |
                        static_cast<unsigned>(yc) << 16 | static_cast<unsigned>(xc);
                    if (key < t[K - 1]) insert_sorted<K>(t, key);
                }
                // The other buffer is written next: the barrier above already
                // ordered every read of it (two displacements ago) before.
                buf ^= 1;
                ++dy;
            }
        }
    }
    if (!live) return;
    DeviceMatch* dst = out + static_cast<long long>(ref - ref_begin) * g.group;
    dst[0] = DeviceMatch{x0, y0, 0, 0.f};
    int filled = 0;
#pragma unroll
    for (int i = 0; i < K; ++i) {
        if (i < g.group - 1 && t[i] != kEmpty) {  // K is group - 1 rounded up to 2^k - 1
            const unsigned at = static_cast<unsigned>(t[i]), bits = static_cast<unsigned>(t[i] >> 32);
            dst[1 + i] = DeviceMatch{static_cast<int>(at & 0xffffu), static_cast<int>(at >> 16), 0,
                                     __uint_as_float(bits)};
            ++filled;
        }
    }
    counts[ref - ref_begin] = 1 + filled;
}

// One batch's launch: its passes, blocks and references per block column.
struct Plan {
    Passes passes;
    int blocks;
    int rows;
};

Plan plan_square(int q, int r, const RasterGrid& grid, int ref_begin, int ref_count) {
    const int refs = kSide - r + 1;
    const int row_first = ref_begin / grid.nx, row_last = (ref_begin + ref_count - 1) / grid.nx;
    const bool clamp_x = grid.max_x % q != 0, clamp_y = grid.max_y % q != 0;
    const int lattice_cols = clamp_x ? grid.nx - 1 : grid.nx;  // references at x = q j
    const int lattice_rows = clamp_y ? grid.ny - 1 : grid.ny;
    const int blocks_x = (lattice_cols + refs - 1) / refs;
    // Bands of at most 16 - R + 1 lattice rows from the batch's first row of
    // y = q i, as even as they go: a batch of a few rows computes only those
    // rows and the R - 1 extra rows their tiles need.
    const int last_lattice = std::min(row_last, lattice_rows - 1);
    const int span = last_lattice - row_first + 1;
    const int bands = span > 0 ? (span + refs - 1) / refs : 0;
    Plan plan{};
    plan.rows = bands > 0 ? (span + bands - 1) / bands : 1;
    // The clamped column (x = max_x) and row (y = max_y) are the last of a
    // lattice shifted so that they fall on it (the last reference thread of
    // its block's row or column).
    const int shift_x = grid.max_x - q * (refs - 1), shift_y = grid.max_y - q * (plan.rows - 1);
    const bool last_row = clamp_y && row_last == grid.ny - 1;
    const auto add = [&](int ox, int oy, int bx, int nbands) {
        if (bx <= 0 || nbands <= 0) return;
        plan.passes.pass[plan.passes.count++] = Pass{ox, oy, bx, plan.blocks};
        plan.blocks += bx * nbands;
    };
    if (blocks_x > 0) add(0, q * row_first, blocks_x, bands);
    if (clamp_x) add(shift_x, q * row_first, 1, bands);
    if (last_row) {
        if (blocks_x > 0) add(0, shift_y, blocks_x, 1);
        if (clamp_x) add(shift_x, shift_y, 1, 1);
    }
    return plan;
}

template <int R, int Q, int K>
void launch_square(const Source& source, const MatchGeometry& g, const RasterGrid& grid, int ref_begin, int ref_count,
                   const Plan& plan, DeviceMatch* out, int* counts, cudaStream_t stream) {
    square_match_kernel<R, Q, K><<<plan.blocks, kSide * (plan.rows + R - 1), 0, stream>>>(
        source, g, grid, ref_begin, ref_count, plan.rows, plan.passes, out, counts);
    NSS_CUDA_CHECK_LAUNCH();
}

template <int R, int Q>
void dispatch_square(const Source& source, const MatchGeometry& g, const RasterGrid& grid, int ref_begin, int ref_count,
                     const Plan& plan, DeviceMatch* out, int* counts, cudaStream_t stream) {
    const int wanted = g.group - 1;
    if (wanted <= 1) {
        launch_square<R, Q, 1>(source, g, grid, ref_begin, ref_count, plan, out, counts, stream);
    } else if (wanted <= 3) {
        launch_square<R, Q, 3>(source, g, grid, ref_begin, ref_count, plan, out, counts, stream);
    } else if (wanted <= 7) {
        launch_square<R, Q, 7>(source, g, grid, ref_begin, ref_count, plan, out, counts, stream);
    } else if (wanted <= 15) {
        launch_square<R, Q, 15>(source, g, grid, ref_begin, ref_count, plan, out, counts, stream);
    } else {
        launch_square<R, Q, 31>(source, g, grid, ref_begin, ref_count, plan, out, counts, stream);
    }
}

int multiprocessors() {
    int device = 0, sms = 0;
    NSS_CUDA_CHECK(cudaGetDevice(&device));
    NSS_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    return sms;
}

}  // namespace

bool spatial_match_square_shape(const MatchGeometry& g, const RasterGrid& grid) {
    const int q = grid.step;
    const bool quarters = (q == 1 || q == 2 || q == 4 || q == 6 || q == 8) && g.block == 2 * q;
    const bool generalized = (g.block == 12 && q == 3) || (g.block == 12 && q == 4) ||
                             (g.block == 16 && q == 4) || (g.block == 8 && q == 2);
    return (quarters || generalized) && g.channels == 1 && g.range_hi < 0 && g.group >= 2 && g.group <= 32 &&
           g.width >= g.block && g.height >= g.block && g.width <= 0xffff && g.height <= 0xffff && grid.nx >= 1 &&
           grid.ny >= 1;
}

bool spatial_match_square_takes(const MatchGeometry& g, const RasterGrid& grid) {
    if (!spatial_match_square_shape(g, grid)) return false;
    // Admission (1080p gray, RTX 5080, 2026-10-28; lane kernels / square,
    // two planes): windows of radius 1 with groups of 4 to 16 and of radius 2
    // with groups of 8 to 16 stay on the lane kernels (square 0.35-0.97x),
    // except at block 12, whose lane kernels are slower (0.95-1.05x there).
    // Everywhere else the square matcher wins up to 4.2x or is within 10%
    // (radius 3 at block 2, groups of 32 at radius 2-3 at block 4).
    const int r = g.lo();
    const bool lanes = grid.step != 6 && g.group >= 4 && g.group <= 16 && (r == 1 || (r == 2 && g.group >= 8));
    return !lanes;
}

namespace {

bool run_square(const Source& source, const MatchGeometry& g, const RasterGrid& grid, int ref_begin, int ref_count,
                DeviceMatch* out, int* counts, cudaStream_t stream, bool any_shape) {
    if (!(any_shape ? spatial_match_square_shape(g, grid) : spatial_match_square_takes(g, grid))) return false;
    if (ref_count <= 0) return true;
    const int r = g.block / grid.step;
    const Plan plan = plan_square(grid.step, r, grid, ref_begin, ref_count);
    // A batch of fewer blocks than the device has SMs leaves it idle: the
    // lane kernels are faster then (1080p gray, RTX 5080: square 0.35-0.97x
    // below 84 blocks, 0.88-1.95x at 80-128, 1.15-2.5x above).
    if (!any_shape && plan.blocks < multiprocessors()) return false;
    if (plan.blocks == 0) return true;
    if (r == 2) {
        switch (grid.step) {
        case 1: dispatch_square<2, 1>(source, g, grid, ref_begin, ref_count, plan, out, counts, stream); break;
        case 2: dispatch_square<2, 2>(source, g, grid, ref_begin, ref_count, plan, out, counts, stream); break;
        case 4: dispatch_square<2, 4>(source, g, grid, ref_begin, ref_count, plan, out, counts, stream); break;
        case 6: dispatch_square<2, 6>(source, g, grid, ref_begin, ref_count, plan, out, counts, stream); break;
        default: dispatch_square<2, 8>(source, g, grid, ref_begin, ref_count, plan, out, counts, stream); break;
        }
    } else if (g.block == 12 && grid.step == 3) {
        dispatch_square<4, 3>(source, g, grid, ref_begin, ref_count, plan, out, counts, stream);
    } else if (g.block == 12 && grid.step == 4) {
        dispatch_square<3, 4>(source, g, grid, ref_begin, ref_count, plan, out, counts, stream);
    } else if (g.block == 16) {
        dispatch_square<4, 4>(source, g, grid, ref_begin, ref_count, plan, out, counts, stream);
    } else {
        dispatch_square<4, 2>(source, g, grid, ref_begin, ref_count, plan, out, counts, stream);
    }
    return true;
}

}  // namespace

bool spatial_match_square(const float* plane, const MatchGeometry& g, const RasterGrid& grid, int ref_begin,
                          int ref_count, DeviceMatch* out, int* counts, cudaStream_t stream, bool any_shape) {
    return run_square(Source{plane, nullptr, 0}, g, grid, ref_begin, ref_count, out, counts, stream, any_shape);
}

bool spatial_match_square_frame(const float* const* frames, int frame, const MatchGeometry& g, const RasterGrid& grid,
                                int ref_begin, int ref_count, DeviceMatch* out, int* counts, cudaStream_t stream,
                                bool any_shape) {
    return run_square(Source{nullptr, frames, frame}, g, grid, ref_begin, ref_count, out, counts, stream, any_shape);
}

}  // namespace nss_cuda
