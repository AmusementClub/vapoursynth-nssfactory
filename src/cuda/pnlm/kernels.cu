// SPDX-License-Identifier: GPL-2.0-only
// PNLM model version 1 on the device (include/nss/params/pnlm.hpp). Built
// without contraction (NSS_CUDA_EXACT_SOURCES): the arithmetic is the CPU
// kernel's (src/cpu/pnlm/pnlm.cpp), operation for operation, so the output has
// the same bits.
//
// Matching is the square matcher of Davy and Ehret (JRTIP 18, 2021, section
// 4.1.4) for patches of 2q at step q: a block of 16 x 16 threads takes 16 x 16
// lattice positions (every q pixels); each thread computes the distance of
// its own q x q quarter for each displacement (its quarter in registers, the
// compared quarter a ring of q rows in registers, one new row a
// displacement), and the 15 x 15 references of the block total their four
// quarters from shared memory. A reference keeps its best keys sorted in
// registers (insertion only when a key beats the last one). Grids whose last
// column or row is clamped (not on the lattice) take extra passes on a
// lattice shifted to put them on it, all in one launch.
//
// Weighting: L lanes of a warp take a reference (pnlm_lanes), lane l the
// patch pixels l, l + L, ... For small windows (pnlm_fused) the matcher
// block does it from shared memory right after matching; otherwise the
// matcher writes the keys, and a second kernel, with more blocks resident,
// weights them. The gather then forms every output pixel from the estimates
// that cover it, in reference order (deterministic, no atomics).
#include "cuda/pnlm/kernels.hpp"
#include "cuda/runtime/error.hpp"

#include <algorithm>

namespace nss_cuda {
namespace {

using nss::PnlmScale;

constexpr int kSide = 16;
constexpr int kRefs = kSide - 1;
constexpr int kSlots = kRefs * kRefs;
constexpr int kThreads = kSide * kSide;
constexpr unsigned kInfinity = 0x7f800000u;
constexpr unsigned long long kEmpty = ~0ull;

struct Pass {
    int ox, oy, blocks_x, by_first, first_block;
};
struct Passes {
    Pass pass[4];
    int count;
};

template <int K>
__device__ __forceinline__ void insert_sorted(unsigned long long (&t)[K], unsigned long long key) {
#pragma unroll
    for (int i = K - 1; i > 0; --i) t[i] = key < t[i - 1] ? t[i - 1] : (key < t[i] ? key : t[i]);
    t[0] = key < t[0] ? key : t[0];
}

// Grid index of the reference at v along an axis, or -1 when none is there.
template <int Q>
__device__ __forceinline__ int grid_index(int v, int max_v, int count) {
    if (v < 0 || v > max_v) return -1;
    if (v == max_v) return count - 1;
    return v % Q == 0 ? v / Q : -1;
}

// The estimate of one reference (k matches: positions xy, weights w) by the L
// lanes of its group; lane ll. Every lane of the warp calls it (the butterfly
// shuffles span the warp); dst null: nothing is stored.
template <int Q>
__device__ __forceinline__ void weigh(int k, const unsigned* xy, const float* w, const PnlmPlane& pl, float* dst,
                                      int ll) {
    constexpr int P = 2 * Q, A = P * P, L = nss::pnlm_lanes(A), PPL = (A + L - 1) / L;
    float acc[PPL], s1 = 0.f, s2 = 0.f, wsum = 0.f;
    int offset[PPL];
    bool on[PPL];
#pragma unroll
    for (int j = 0; j < PPL; ++j) {
        const int p = ll + j * L;
        on[j] = p < A;
        offset[j] = (p / P) * pl.pitch + p % P;
        acc[j] = 0.f;
    }
#pragma unroll 4
    for (int i = 0; i < k; ++i) {
        const unsigned pos = xy[i];
        const float wt = w[i];
        wsum = wsum + wt;
        const float* at = pl.frames[nss::pnlm_pos_t(pos)] + static_cast<long long>(nss::pnlm_pos_y(pos)) * pl.pitch +
                          nss::pnlm_pos_x(pos);
#pragma unroll
        for (int j = 0; j < PPL; ++j) {
            if (!on[j]) continue;
            const float v = __ldg(at + offset[j]);
            acc[j] = fmaf(wt, v, acc[j]);
            s1 = s1 + v;
            s2 = fmaf(v, v, s2);
        }
    }
    __syncwarp();
#pragma unroll
    for (int o = L / 2; o > 0; o >>= 1) {
        s1 = s1 + __shfl_xor_sync(0xffffffffu, s1, o);
        s2 = s2 + __shfl_xor_sync(0xffffffffu, s2, o);
    }
    if (k == 0 || !dst) return;
    const nss::PnlmFlat flat = nss::pnlm_flat(s1, s2, static_cast<float>(k * A), pl.scale);
#pragma unroll
    for (int j = 0; j < PPL; ++j) {
        if (on[j]) dst[ll + j * L] = nss::pnlm_mix(acc[j] / wsum, flat);
    }
}

// MODE 1: keys and counts out. MODE 2: fused weighting into est.
template <int Q, int K, int MODE, int MINB>
__global__ void __launch_bounds__(kThreads, MINB)
    match_kernel(PnlmPlane pl, int nx, int ny, Passes passes, unsigned long long* keys, int* counts, float* est) {
    constexpr int P = 2 * Q, A = P * P;
    constexpr int FS = MODE == 2 ? kSlots : 1;
    __shared__ float part[2][kSide][kSide + 1];
    __shared__ unsigned sxy[FS][K + 1];
    __shared__ float sw[FS][K + 1];
    __shared__ int sk[FS], sref[FS];
    const int tid = threadIdx.x, tx = tid % kSide, ty = tid / kSide;
    int which = 0;
    for (int i = 1; i < passes.count; ++i) {
        if (static_cast<int>(blockIdx.x) >= passes.pass[i].first_block) which = i;
    }
    const Pass& pass = passes.pass[which];
    const int local = static_cast<int>(blockIdx.x) - pass.first_block;
    const int bx = local % pass.blocks_x;
    const int by = pass.by_first + local / pass.blocks_x;
    const int x0 = pass.ox + Q * (kRefs * bx + tx);
    const int y0 = pass.oy + Q * (kRefs * by + ty);
    const int max_x = pl.width - P, max_y = pl.height - P, last_row = pl.height - 1, last_col = pl.width - 1;
    const int col = grid_index<Q>(x0, max_x, nx), row = grid_index<Q>(y0, max_y, ny);
    const int ref = row * nx + col;
    const bool live = tx < kRefs && ty < kRefs && col >= 0 && row >= 0;
    const int a = pl.a;
    const long long pitch = pl.pitch;
    const float* const center = pl.frames[pl.center];
    int cols[Q];
#pragma unroll
    for (int c = 0; c < Q; ++c) cols[c] = min(max(x0 + c, 0), last_col);
    float rq[Q][Q];
#pragma unroll
    for (int r = 0; r < Q; ++r) {
        const long long at = min(max(y0 + r, 0), last_row) * pitch;
#pragma unroll
        for (int c = 0; c < Q; ++c) rq[r][c] = center[at + cols[c]];
    }
    unsigned long long best[K];
#pragma unroll
    for (int i = 0; i < K; ++i) best[i] = kEmpty;
    int buf = 0;
    for (int t = 0; t < pl.count; ++t) {
        const float* const plane = pl.frames[t];
        const bool other = t != pl.center;
        for (int dx = -a; dx <= a; ++dx) {
            int xs[Q];
#pragma unroll
            for (int c = 0; c < Q; ++c) xs[c] = min(max(x0 + c + dx, 0), last_col);
            // v[slot][column]: before phase s, row y0 + dy + k is in slot (s + 1 + k) % Q for k < Q - 1.
            float v[Q][Q];
#pragma unroll
            for (int k = 0; k < Q - 1; ++k) {
                const long long at = min(max(y0 - a + k, 0), last_row) * pitch;
#pragma unroll
                for (int c = 0; c < Q; ++c) v[k + 1][c] = plane[at + xs[c]];
            }
            const int xc = x0 + dx;
            const bool col_ok = live && xc >= 0 && xc <= max_x;
            int dy = -a;
            while (dy <= a) {
#pragma unroll
                for (int s = 0; s < Q; ++s) {
                    if (dy > a) break;  // uniform over the block
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
                    part[buf][ty][tx] = nss::pnlm_columns<Q>(cs);
                    __syncthreads();
                    const int yc = y0 + dy;
                    if (col_ok && yc >= 0 && yc <= max_y && ((dx | dy) != 0 || other)) {
                        const float sum = (part[buf][ty][tx] + part[buf][ty][tx + 1]) +
                                          (part[buf][ty + 1][tx] + part[buf][ty + 1][tx + 1]);
                        const unsigned long long key =
                            static_cast<unsigned long long>(min(__float_as_uint(sum), kInfinity)) << 32 |
                            nss::pnlm_pos(t, yc, xc);
                        if (key < best[K - 1]) insert_sorted<K>(best, key);
                    }
                    // The other buffer is written next: the barrier above
                    // ordered every read of it (two displacements ago) before.
                    buf ^= 1;
                    ++dy;
                }
            }
        }
    }
    const int group = pl.group;
    if constexpr (MODE == 1) {
        if (!live) return;
        const long long base = static_cast<long long>(ref) * group;
        keys[base] = nss::pnlm_pos(pl.center, y0, x0);
        int filled = 0;
#pragma unroll
        for (int i = 0; i < K; ++i) {
            if (i < group - 1 && best[i] != kEmpty) {  // K is group - 1 rounded up to 2^k - 1
                keys[base + 1 + i] = best[i];
                ++filled;
            }
        }
        counts[ref] = 1 + filled;
    } else {
        if (tx < kRefs && ty < kRefs) {
            const int slot = ty * kRefs + tx;
            if (!live) {
                sk[slot] = 0;
            } else {
                sref[slot] = ref;
                sxy[slot][0] = nss::pnlm_pos(pl.center, y0, x0);
                sw[slot][0] = 1.f;
                int n = 1;
#pragma unroll
                for (int i = 0; i < K; ++i) {
                    if (i < group - 1 && best[i] != kEmpty) {
                        sxy[slot][n] = static_cast<unsigned>(best[i]);
                        sw[slot][n] = nss::pnlm_weight(__uint_as_float(static_cast<unsigned>(best[i] >> 32)), pl.scale);
                        ++n;
                    }
                }
                sk[slot] = n;
            }
        }
        __syncthreads();
        constexpr int L = nss::pnlm_lanes(A), SPW = 32 / L;
        const int warp = tid >> 5, lane = tid & 31, ll = lane % L;
        for (int b = warp * SPW; b < kSlots; b += (kThreads / 32) * SPW) {
            const int slot = min(b + lane / L, kSlots - 1);
            const int k = b + lane / L < kSlots ? sk[slot] : 0;
            weigh<Q>(k, sxy[slot], sw[slot], pl, k ? est + static_cast<long long>(sref[slot]) * A : nullptr, ll);
        }
    }
}

// The keys path's weighting: 256 threads take 256 / L references.
template <int Q>
__global__ void __launch_bounds__(256) weight_kernel(PnlmPlane pl, int refs, const unsigned long long* keys,
                                                     const int* counts, float* est) {
    constexpr int P = 2 * Q, A = P * P, L = nss::pnlm_lanes(A), R = 256 / L;
    __shared__ unsigned sxy[R][nss::kPnlmMaxGroup];
    __shared__ float sw[R][nss::kPnlmMaxGroup];
    __shared__ int sk[R];
    const int tid = threadIdx.x, r0 = blockIdx.x * R, group = pl.group;
    for (int j = tid; j < R * group; j += 256) {
        const int s = j / group, i = j % group, ref = r0 + s;
        if (ref >= refs) {
            if (i == 0) sk[s] = 0;
            continue;
        }
        const int k = counts[ref];
        if (i == 0) sk[s] = k;
        if (i < k) {
            const unsigned long long key = keys[static_cast<long long>(ref) * group + i];
            sxy[s][i] = static_cast<unsigned>(key);
            sw[s][i] = i == 0 ? 1.f : nss::pnlm_weight(__uint_as_float(static_cast<unsigned>(key >> 32)), pl.scale);
        }
    }
    __syncthreads();
    const int s = tid / L, ll = tid % L, k = sk[s];
    weigh<Q>(k, sxy[s], sw[s], pl, k ? est + static_cast<long long>(r0 + s) * A : nullptr, ll);
}

// Grid entries covering v along an axis (see the CPU kernel's cover()).
struct Cover {
    int lo, hi, last;
};
__device__ __forceinline__ Cover cover(int v, int n, int count, int p) {
    const int q = p / 2;
    Cover c;
    c.lo = v - p + 1 <= 0 ? 0 : (v - p + 1 + q - 1) / q;
    c.hi = min(count - 1, v / q);
    const int last = n - p;
    c.last = (c.hi < count - 1 && last <= v && v < last + p) ? count - 1 : -1;
    return c;
}

// Every output pixel: the tent-weighted mean of the estimates covering it.
__global__ void gather_kernel(const float* est, int nx, int ny, int p, int width, int height, int pitch, float* out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const Cover cx = cover(x, width, nx, p), cy = cover(y, height, ny, p);
    const int area = p * p;
    float n = 0.f, d = 0.f;
    for (int ia = 0; ia <= cy.hi - cy.lo + (cy.last >= 0); ++ia) {
        const int iy = cy.lo + ia <= cy.hi ? cy.lo + ia : cy.last;
        const int gy = nss::pnlm_grid_pos(iy, height, p);
        if (gy > y || y >= gy + p) continue;
        const float wy = nss::pnlm_tent(y - gy, p);
        for (int ib = 0; ib <= cx.hi - cx.lo + (cx.last >= 0); ++ib) {
            const int ix = cx.lo + ib <= cx.hi ? cx.lo + ib : cx.last;
            const int gx = nss::pnlm_grid_pos(ix, width, p);
            if (gx > x || x >= gx + p) continue;
            const float wxy = wy * nss::pnlm_tent(x - gx, p);
            d = d + wxy;
            n = fmaf(wxy, est[(static_cast<long long>(iy) * nx + ix) * area + (y - gy) * p + (x - gx)], n);
        }
    }
    out[static_cast<long long>(y) * pitch + x] = n / d;
}

template <int Q, int K, int MODE>
void launch_match(const PnlmPlane& pl, int nx, int ny, unsigned long long* keys, int* counts, float* est,
                  cudaStream_t stream) {
    // Measured on an RTX 5080 (1080p gray): three blocks an SM help q = 3;
    // four the key path of q = 2; q = 4 spills under a bound.
    constexpr int MINB = Q == 3 ? 3 : (Q == 2 && MODE == 1 ? 4 : 1);
    const int max_x = pl.width - 2 * Q, max_y = pl.height - 2 * Q;
    const bool clamp_x = max_x % Q != 0, clamp_y = max_y % Q != 0;
    const int lattice_cols = clamp_x ? nx - 1 : nx;
    const int lattice_rows = clamp_y ? ny - 1 : ny;
    const int blocks_x = (lattice_cols + kRefs - 1) / kRefs;
    const int bands = (lattice_rows + kRefs - 1) / kRefs;
    // The clamped column (x = max_x) and row are the 15th of a lattice shifted
    // so that they fall on it (thread 14 of its block).
    const int shift_x = max_x - Q * (kRefs - 1), shift_y = max_y - Q * (kRefs - 1);
    Passes passes{};
    int blocks = 0;
    const auto add = [&](int ox, int oy, int bx, int by_first, int nbands) {
        if (bx <= 0 || nbands <= 0) return;
        passes.pass[passes.count++] = Pass{ox, oy, bx, by_first, blocks};
        blocks += bx * nbands;
    };
    if (blocks_x > 0) add(0, 0, blocks_x, 0, bands);
    if (clamp_x) add(shift_x, 0, 1, 0, bands);
    if (clamp_y) {
        if (blocks_x > 0) add(0, shift_y, blocks_x, 0, 1);
        if (clamp_x) add(shift_x, shift_y, 1, 0, 1);
    }
    if (blocks == 0) return;
    match_kernel<Q, K, MODE, MINB><<<blocks, kThreads, 0, stream>>>(pl, nx, ny, passes, keys, counts, est);
    NSS_CUDA_CHECK_LAUNCH();
}

template <int Q>
void run_q(const PnlmPlane& pl, float* est, void* scratch, cudaStream_t stream) {
    const int nx = nss::pnlm_grid_count(pl.width, 2 * Q), ny = nss::pnlm_grid_count(pl.height, 2 * Q);
    const int refs = nx * ny;
    auto* keys = static_cast<unsigned long long*>(scratch);
    int* counts = keys ? reinterpret_cast<int*>(keys + static_cast<std::size_t>(refs) * pl.group) : nullptr;
    if (pnlm_fused(pl.a, pl.group)) {
        if (pl.group <= 8) launch_match<Q, 7, 2>(pl, nx, ny, nullptr, nullptr, est, stream);
        else launch_match<Q, 15, 2>(pl, nx, ny, nullptr, nullptr, est, stream);
    } else {
        if (pl.group <= 8) launch_match<Q, 7, 1>(pl, nx, ny, keys, counts, nullptr, stream);
        else if (pl.group <= 16) launch_match<Q, 15, 1>(pl, nx, ny, keys, counts, nullptr, stream);
        else launch_match<Q, 31, 1>(pl, nx, ny, keys, counts, nullptr, stream);
        constexpr int R = 256 / nss::pnlm_lanes(4 * Q * Q);
        weight_kernel<Q><<<(refs + R - 1) / R, 256, 0, stream>>>(pl, refs, keys, counts, est);
        NSS_CUDA_CHECK_LAUNCH();
    }
}

}  // namespace

std::size_t pnlm_est_bytes(int width, int height, int block) {
    return static_cast<std::size_t>(nss::pnlm_grid_count(width, block)) * nss::pnlm_grid_count(height, block) * block *
           block * sizeof(float);
}

std::size_t pnlm_keys_bytes(int width, int height, int block, int group) {
    const std::size_t refs =
        static_cast<std::size_t>(nss::pnlm_grid_count(width, block)) * nss::pnlm_grid_count(height, block);
    return refs * group * sizeof(unsigned long long) + refs * sizeof(int);
}

bool pnlm_fused(int a, int group) { return a <= 4 && group <= 16; }

void pnlm_run(const PnlmPlane& plane, float* est, void* keys, float* out, cudaStream_t stream) {
    switch (plane.block) {
    case 4: run_q<2>(plane, est, keys, stream); break;
    case 6: run_q<3>(plane, est, keys, stream); break;
    case 8: run_q<4>(plane, est, keys, stream); break;
    default: return;
    }
    const int nx = nss::pnlm_grid_count(plane.width, plane.block), ny = nss::pnlm_grid_count(plane.height, plane.block);
    const dim3 t(32, 8), b((plane.width + 31) / 32, (plane.height + 7) / 8);
    gather_kernel<<<b, t, 0, stream>>>(est, nx, ny, plane.block, plane.width, plane.height, plane.pitch, out);
    NSS_CUDA_CHECK_LAUNCH();
}

}  // namespace nss_cuda
