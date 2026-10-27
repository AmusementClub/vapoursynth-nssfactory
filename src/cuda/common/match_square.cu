// SPDX-License-Identifier: GPL-2.0-only
// Square matcher for block 8 at step 4 (Davy & Ehret, "GPU acceleration of
// NL-means, BM3D and VBM3D", JRTIP 18 (2021), section 4.1.4).
//
// At step 4 an 8 x 8 block is four 4 x 4 quarters, and each quarter is the
// top-left quarter of a reference of the grid: the quarter distances for a
// displacement are shared by the four references that contain them. A block
// of 16 x 16 threads takes 16 x 16 lattice positions; every thread computes
// the distance of its own top-left quarter for each displacement (its
// reference quarter in registers, the compared quarter a ring of four rows in
// registers, one new row a displacement), and the 15 x 15 references of the
// block's top-left corner total their four quarters from shared memory.
// A reference keeps its best group - 1 keys sorted in registers (the paper's
// Algorithm 2: an insertion only when a key beats the last one).
//
// Keys and order are the lane kernels' (distance bits capped at +inf, then
// y, then x; the reference first), but a distance is the sum of four quarter
// sums (4-row column chains, then a tree) instead of the lane kernels' 8-row
// chains: the last bits differ, so near ties may resolve differently.
//
// Grids whose last column or row is clamped (max_x or max_y not a multiple of
// 4) take extra passes on a lattice shifted to put the clamped references on
// it; a reference is on exactly one pass's lattice.
#include "cuda/common/match.hpp"
#include "cuda/runtime/error.hpp"

#include <algorithm>

namespace nss_cuda {
namespace {

constexpr int kSide = 16;         // threads per block side
constexpr int kRefs = kSide - 1;  // references per block side
constexpr unsigned kInfinity = 0x7f800000u;
constexpr unsigned long long kEmpty = ~0ull;

template <int K>
__device__ __forceinline__ void insert_sorted(unsigned long long (&t)[K], unsigned long long key) {
#pragma unroll
    for (int i = K - 1; i > 0; --i) t[i] = key < t[i - 1] ? t[i - 1] : (key < t[i] ? key : t[i]);
    t[0] = key < t[0] ? key : t[0];
}

// A lattice: position (j, i) of block (bx, by) is the quarter at
// (ox + 4 (15 bx + j), oy + 4 (15 by + i)).
struct Pass {
    int ox, oy;
    int blocks_x;     // blocks per lattice row band
    int by_first;     // first band
    int first_block;  // first block of the launch that runs this pass
};
// The passes of one batch in one launch: the main lattice and up to three
// shifted ones for clamped edges (separate launches of a few blocks each cost
// a whole block's time apiece on small frames).
struct Passes {
    Pass pass[4];
    int count;
};

// Grid index of the reference at x (y), or -1 when no reference is there.
__device__ __forceinline__ int grid_index(int v, int max_v, int count) {
    if (v < 0 || v > max_v) return -1;
    if (v == max_v) return count - 1;
    return (v & 3) == 0 ? v >> 2 : -1;
}

template <int K>
__global__ void __launch_bounds__(kSide * kSide) square_match_kernel(const float* plane, MatchGeometry g,
                                                                     RasterGrid grid, int ref_begin, int ref_count,
                                                                     Passes passes, DeviceMatch* out, int* counts) {
    __shared__ float part[2][kSide][kSide + 1];
    const int tid = threadIdx.x, tx = tid % kSide, ty = tid / kSide;
    int which = 0;
    for (int i = 1; i < passes.count; ++i) {
        if (static_cast<int>(blockIdx.x) >= passes.pass[i].first_block) which = i;
    }
    const Pass& pass = passes.pass[which];
    const int local = static_cast<int>(blockIdx.x) - pass.first_block;
    const int bx = local % pass.blocks_x;
    const int by = pass.by_first + local / pass.blocks_x;
    const int x0 = pass.ox + 4 * (kRefs * bx + tx);
    const int y0 = pass.oy + 4 * (kRefs * by + ty);
    const int col = grid_index(x0, grid.max_x, grid.nx), row = grid_index(y0, grid.max_y, grid.ny);
    const int ref = row * grid.nx + col;
    const bool live = tx < kRefs && ty < kRefs && col >= 0 && row >= 0 && ref >= ref_begin &&
                      ref < ref_begin + ref_count;
    const int lo = g.lo(), hi = g.hi();
    const int max_x = g.width - 8, max_y = g.height - 8, last_row = g.height - 1, last_col = g.width - 1;
    const long long pitch = g.pitch;
    int cols[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) cols[c] = min(max(x0 + c, 0), last_col);
    float rq[4][4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const long long at = min(max(y0 + r, 0), last_row) * pitch;
#pragma unroll
        for (int c = 0; c < 4; ++c) rq[r][c] = plane[at + cols[c]];
    }
    unsigned long long t[K];
#pragma unroll
    for (int i = 0; i < K; ++i) t[i] = kEmpty;
    int buf = 0;
    for (int dx = -lo; dx <= hi; ++dx) {
        int xs[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) xs[c] = min(max(x0 + c + dx, 0), last_col);
        // v[slot][column]: before phase s, row y0 + dy + k is in slot (s + 1 + k) % 4 for k < 3.
        float v[4][4];
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            const long long at = min(max(y0 - lo + k, 0), last_row) * pitch;
#pragma unroll
            for (int c = 0; c < 4; ++c) v[k + 1][c] = plane[at + xs[c]];
        }
        const int xc = x0 + dx;
        const bool col_ok = live && xc >= 0 && xc <= max_x;
        int dy = -lo;
        while (dy <= hi) {
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                if (dy > hi) break;  // uniform over the block
                const long long at = min(max(y0 + dy + 3, 0), last_row) * pitch;
#pragma unroll
                for (int c = 0; c < 4; ++c) v[s][c] = plane[at + xs[c]];
                float cs[4];
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    float sum = 0.f;
#pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        const float d = rq[r][c] - v[(s + 1 + r) % 4][c];
                        sum = fmaf(d, d, sum);
                    }
                    cs[c] = sum;
                }
                part[buf][ty][tx] = (cs[0] + cs[1]) + (cs[2] + cs[3]);
                __syncthreads();
                const int yc = y0 + dy;
                if (col_ok && yc >= 0 && yc <= max_y && (dx | dy) != 0) {
                    const float sum = (part[buf][ty][tx] + part[buf][ty][tx + 1]) +
                                      (part[buf][ty + 1][tx] + part[buf][ty + 1][tx + 1]);
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

template <int K>
void launch_square(const float* plane, const MatchGeometry& g, const RasterGrid& grid, int ref_begin, int ref_count,
                   DeviceMatch* out, int* counts, cudaStream_t stream) {
    const int row_first = ref_begin / grid.nx, row_last = (ref_begin + ref_count - 1) / grid.nx;
    const bool clamp_x = grid.max_x % 4 != 0, clamp_y = grid.max_y % 4 != 0;
    const int lattice_cols = clamp_x ? grid.nx - 1 : grid.nx;  // references at x = 4 j
    const int lattice_rows = clamp_y ? grid.ny - 1 : grid.ny;
    const int blocks_x = (lattice_cols + kRefs - 1) / kRefs;
    // Bands of 15 lattice rows holding the batch's rows of y = 4 i.
    const int last_lattice = std::min(row_last, lattice_rows - 1);
    const int band_first = row_first / kRefs;
    const int bands = row_first <= last_lattice ? last_lattice / kRefs - band_first + 1 : 0;
    // The clamped column (x = max_x) and row (y = max_y) are the 15th of a
    // lattice shifted so that they fall on it (thread 14 of its block).
    const int shift_x = grid.max_x - 4 * (kRefs - 1), shift_y = grid.max_y - 4 * (kRefs - 1);
    const bool last_row = clamp_y && row_last == grid.ny - 1;
    Passes passes{};
    int blocks = 0;
    const auto add = [&](int ox, int oy, int bx, int by_first, int nbands) {
        if (bx <= 0 || nbands <= 0) return;
        passes.pass[passes.count++] = Pass{ox, oy, bx, by_first, blocks};
        blocks += bx * nbands;
    };
    if (blocks_x > 0) add(0, 0, blocks_x, band_first, bands);
    if (clamp_x) add(shift_x, 0, 1, band_first, bands);
    if (last_row) {
        if (blocks_x > 0) add(0, shift_y, blocks_x, 0, 1);
        if (clamp_x) add(shift_x, shift_y, 1, 0, 1);
    }
    if (blocks == 0) return;
    square_match_kernel<K><<<blocks, kSide * kSide, 0, stream>>>(plane, g, grid, ref_begin, ref_count, passes, out,
                                                                  counts);
    NSS_CUDA_CHECK_LAUNCH();
}

}  // namespace

bool spatial_match_square_takes(const MatchGeometry& g, const RasterGrid& grid) {
    if (!(g.block == 8 && grid.step == 4 && g.channels == 1 && g.range_hi < 0 && g.group >= 2 && g.group <= 32 &&
          g.width >= 8 && g.height >= 8 && g.width <= 0xffff && g.height <= 0xffff && grid.nx >= 1 && grid.ny >= 1)) {
        return false;
    }
    // Admission (1080p gray, RTX 5080, 2026-10-27; lane kernels / square):
    // windows of radius 1-2 with groups of 8 or more stay on the lane kernels
    // (0.55-0.95x); everywhere else the square matcher wins, 1.07-3.5x.
    return !(g.lo() <= 2 && g.group >= 8);
}

bool spatial_match_square(const float* plane, const MatchGeometry& g, const RasterGrid& grid, int ref_begin,
                          int ref_count, DeviceMatch* out, int* counts, cudaStream_t stream) {
    if (!spatial_match_square_takes(g, grid)) return false;
    if (ref_count <= 0) return true;
    const int wanted = g.group - 1;
    if (wanted <= 1) {
        launch_square<1>(plane, g, grid, ref_begin, ref_count, out, counts, stream);
    } else if (wanted <= 3) {
        launch_square<3>(plane, g, grid, ref_begin, ref_count, out, counts, stream);
    } else if (wanted <= 7) {
        launch_square<7>(plane, g, grid, ref_begin, ref_count, out, counts, stream);
    } else if (wanted <= 15) {
        launch_square<15>(plane, g, grid, ref_begin, ref_count, out, counts, stream);
    } else {
        launch_square<31>(plane, g, grid, ref_begin, ref_count, out, counts, stream);
    }
    return true;
}

}  // namespace nss_cuda
