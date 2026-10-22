// SPDX-License-Identifier: GPL-2.0-only
// nss_cuda.DA3D: the CPU model (include/nss/params/da3d.hpp, src/cpu/da3d)
// operation for operation. One block per tile walks the tile's greedy patch
// loop; every 32-lane sum is one warp's chain (lane l adds the terms l,
// l + 32, ... in order, then the halving tree), the radix-4 64 x 64 DFT runs
// in shared memory with the CPU's butterflies, and every fused multiply-add
// is explicit (this TU is built with --fmad=false). The tiles' sums are added
// in tile order by a second kernel.
#include "cuda/da3d/kernels.hpp"
#include "cuda/runtime/error.hpp"
#include "nss/params/da3d.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <cstdio>

namespace nss_cuda {
namespace {

constexpr int kS = nss::kDa3dPatch;
constexpr int kR = nss::kDa3dRadius;
constexpr int kArea = kS * kS;
constexpr int kAfter = kS - kR - 1;
constexpr int kThreads = 512;
// Shared-memory rows of the transform padded to 65 floats: a warp reading one
// column of 32 rows touches 32 banks.
constexpr int kPitch = 65;
constexpr int kShared = 64 * kPitch;
constexpr int kSharedLevel = 3;            // weight-map levels from here up in shared memory
constexpr int kSharedLevelFloats = 1536;   // levels >= 3 of a 256 x 256 level 0 (1024 + 256 + 64 + 16 + 4 + 4, padded)

constexpr std::size_t align16(std::size_t n) { return (n + 15) & ~std::size_t{15}; }

int next_pow2(int n) {
    --n;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}
int number_of_bits(int n) {
    int bits = 0;
    while (n) {
        ++bits;
        n >>= 1;
    }
    return bits;
}
std::size_t wm_floats(int h, int w) {
    std::size_t total = 0;
    const int lv = std::max(number_of_bits(h - 1) + 1, number_of_bits(w - 1) + 1);
    int rr = next_pow2(h), cr = next_pow2(w);
    for (int l = 0; l < lv; ++l) {
        total += align16(static_cast<std::size_t>(rr) * cr);
        rr = ((rr + 3) >> 2) << 1;
        cr = ((cr + 3) >> 2) << 1;
    }
    return total;
}

__device__ __forceinline__ int dev_next_pow2(int n) {
    --n;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}
__device__ __forceinline__ int dev_bits(int n) { return n ? 32 - __clz(n) : 0; }
__device__ __forceinline__ int symmetric(int pos, int size) {
    if (pos < 0) pos = -pos - 1;
    if (pos >= 2 * size) pos %= 2 * size;
    if (pos >= size) pos = 2 * size - 1 - pos;
    return pos;
}
__device__ __forceinline__ int rev4(int i) { return (i & 3) << 4 | (i & 12) | (i >> 4); }
__device__ __forceinline__ int negrow(int i) { return rev4((64 - rev4(i)) & 63); }

// The reference's exponential (nss::da3d_fastexp).
__device__ __forceinline__ float fastexp(float x) {
    const float t = 12102203.f * x;
    if (!(t > -1065353216.f)) return 0.f;
    int bits = __float2int_rz(t) + 1065353216;
    if (bits <= 0) return 0.f;
    return __int_as_float(bits);
}

struct TileParams {
    const float* plane_noisy[3];
    const float* plane_guide[3];
    int width, height;
    int rows_t, cols_t;
    int tile_w;            // padded tile stride (max tile width + 63)
    std::size_t tile_area; // tile_w x padded max height
    std::size_t wm_stride; // floats of a tile's weight map
    float sigma2;
    const float* twiddle;
    const float* dist_reg;
    const float* dist_k;
    float* tile_num;  // tiles x channels x tile_area
    float* tile_den;  // tiles x tile_area
    float* wm;        // tiles x wm_stride
    float* scratch;   // tiles x 2 C x kArea: the spectra
    unsigned long long* stats;
};

__device__ __forceinline__ int tile_start(int n, int count, int i) {
    return static_cast<int>(static_cast<long long>(n) * i / count);
}

// One warp: the 32-lane sum of term(acc, i) over i in [0, 4096).
template <class Term>
__device__ __forceinline__ float warp_lane_sum(const Term& term) {
    const int lane = threadIdx.x & 31;
    float acc = 0.f;
#pragma unroll 8
    for (int i = lane; i < kArea; i += 32) acc = term(acc, i);
#pragma unroll
    for (int half = 16; half > 0; half >>= 1) {
        const float other = __shfl_down_sync(0xffffffffu, acc, half);
        acc = acc + other;
    }
    return acc;  // lane 0
}

// (xr, xi) times (wr, wi): (xr wr - xi wi, xr wi + xi wr), fused as on the CPU.
__device__ __forceinline__ void twiddle_mul(float& xr, float& xi, float wr, float wi) {
    const float r = fmaf(xr, wr, -(xi * wi));
    xi = fmaf(xr, wi, xi * wr);
    xr = r;
}

// One radix-4 butterfly of the CPU's Dif4 / Dit4 (twiddle index t): x0..x3
// in, outputs in place in the CPU's order.
template <bool Dit>
__device__ __forceinline__ void butterfly(float* xr, float* xi, int t, bool inverse, const float* tw) {
    const float sign = inverse ? -1.f : 1.f;
    const float w1r = tw[t], w1i = sign * tw[64 + t];
    const float w2r = tw[2 * t], w2i = sign * tw[64 + 2 * t];
    const float w3r = tw[3 * t], w3i = sign * tw[64 + 3 * t];
    if (Dit && t != 0) {
        twiddle_mul(xr[1], xi[1], w1r, w1i);
        twiddle_mul(xr[2], xi[2], w2r, w2i);
        twiddle_mul(xr[3], xi[3], w3r, w3i);
    }
    const float ar = xr[0] + xr[2], ai = xi[0] + xi[2];
    const float br = xr[0] - xr[2], bi = xi[0] - xi[2];
    const float cr = xr[1] + xr[3], ci = xi[1] + xi[3];
    const float dr = xr[1] - xr[3], di = xi[1] - xi[3];
    float mr = br + di, mi = bi - dr;  // b - i d
    float pr = br - di, pi = bi + dr;  // b + i d
    if (inverse) {
        const float tr = mr, ti = mi;
        mr = pr;
        mi = pi;
        pr = tr;
        pi = ti;
    }
    float er = ar - cr, ei = ai - ci;
    if (!Dit && t != 0) {
        twiddle_mul(mr, mi, w1r, w1i);
        twiddle_mul(er, ei, w2r, w2i);
        twiddle_mul(pr, pi, w3r, w3i);
    }
    xr[0] = ar + cr;
    xi[0] = ai + ci;
    xr[1] = mr;
    xi[1] = mi;
    xr[2] = er;
    xi[2] = ei;
    xr[3] = pr;
    xi[3] = pi;
}

// The stage of quarter Q (rows g + j + {0, Q, 2Q, 3Q}) along the rows (Rows:
// each column) or along the columns (each row), on threads lt of nt.
template <bool Dit, bool Rows, int Q>
__device__ __forceinline__ void radix4_stage(float* re, float* im, bool inverse, const float* tw, int lt, int nt) {
    constexpr int S = 16 / Q;
    for (int item = lt; item < 1024; item += nt) {
        // Rows: lanes along the columns; columns: lanes along the rows (one
        // butterfly position of 64 rows; the 65-float pitch keeps their banks apart)
        const int line = item & 63;
        const int pair = item >> 6;
        const int j = pair % Q, g = (pair / Q) * 4 * Q;
        constexpr int unit = Rows ? kPitch : 1;
        const int base = Rows ? (g + j) * kPitch + line : line * kPitch + g + j;
        float xr[4], xi[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            xr[u] = re[base + u * Q * unit];
            xi[u] = im[base + u * Q * unit];
        }
        butterfly<Dit>(xr, xi, j * S, inverse, tw);
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            re[base + u * Q * unit] = xr[u];
            im[base + u * Q * unit] = xi[u];
        }
    }
}

// The stages of quarters 16 and 4 together (DIF: 16 then 4; DIT: 4 then
// 16): a task holds the 16 elements j0 + 4a + 16b (a, b < 4) of a line in
// registers; both stages' butterflies stay within them.
template <bool Dit, bool Rows>
__device__ __forceinline__ void radix4_stages_16_4(float* re, float* im, bool inverse, const float* tw, int lt, int nt) {
    for (int task = lt; task < 256; task += nt) {
        const int line = task & 63, j0 = task >> 6;
        constexpr int unit = Rows ? kPitch : 1;
        const int base = Rows ? j0 * kPitch + line : line * kPitch + j0;
        float vr[16], vi[16];  // [a * 4 + b]: position j0 + 4a + 16b
#pragma unroll
        for (int a = 0; a < 4; ++a)
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                vr[a * 4 + b] = re[base + (4 * a + 16 * b) * unit];
                vi[a * 4 + b] = im[base + (4 * a + 16 * b) * unit];
            }
        auto stage16 = [&]() {  // quarter 16: over b, twiddle j0 + 4a
#pragma unroll
            for (int a = 0; a < 4; ++a) {
                float xr[4], xi[4];
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    xr[b] = vr[a * 4 + b];
                    xi[b] = vi[a * 4 + b];
                }
                butterfly<Dit>(xr, xi, j0 + 4 * a, inverse, tw);
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    vr[a * 4 + b] = xr[b];
                    vi[a * 4 + b] = xi[b];
                }
            }
        };
        auto stage4 = [&]() {  // quarter 4: over a, twiddle 4 j0
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                float xr[4], xi[4];
#pragma unroll
                for (int a = 0; a < 4; ++a) {
                    xr[a] = vr[a * 4 + b];
                    xi[a] = vi[a * 4 + b];
                }
                butterfly<Dit>(xr, xi, 4 * j0, inverse, tw);
#pragma unroll
                for (int a = 0; a < 4; ++a) {
                    vr[a * 4 + b] = xr[a];
                    vi[a * 4 + b] = xi[a];
                }
            }
        };
        if (Dit) {
            stage4();
            stage16();
        } else {
            stage16();
            stage4();
        }
#pragma unroll
        for (int a = 0; a < 4; ++a)
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                re[base + (4 * a + 16 * b) * unit] = vr[a * 4 + b];
                im[base + (4 * a + 16 * b) * unit] = vi[a * 4 + b];
            }
    }
}

// Rows p and rev4(p) exchanged (rev4 is its own inverse).
__device__ void swap_rows(float* re, float* im, int lt, int nt) {
    for (int e = lt; e < kArea; e += nt) {
        const int row = e >> 6, other = rev4(row);
        if (row < other) {
            const int i = row * kPitch + (e & 63);
            const int j = other * kPitch + (e & 63);
            const float a = re[i], b = im[i];
            re[i] = re[j];
            im[i] = im[j];
            re[j] = a;
            im[j] = b;
        }
    }
}

// The 64 x 64 transforms as steps between barriers, on threads lt of nt
// (a whole block or half of one):
//   forward, x[row][col] -> F at [rev4 fr][rev4 fc]: radix-4 DIF along the
//   rows (stages 16 and 4 together, then 1), then along the columns;
//   inverse of that layout -> y at [rev4 row][col]: DIT along the columns,
//   the rows swapped by rev4, DIF along the rows.
// The CPU's Forward2 / Inverse2 arithmetic (its row passes on its transposed
// copy), in place.
constexpr int kForwardSteps = 4, kInverseSteps = 5;
__device__ __forceinline__ void transform_step(bool inverse, int step, float* re, float* im, const float* tw, int lt,
                                               int nt) {
    if (!inverse) {
        switch (step) {
            case 0: radix4_stages_16_4<false, true>(re, im, false, tw, lt, nt); break;
            case 1: radix4_stage<false, true, 1>(re, im, false, tw, lt, nt); break;
            case 2: radix4_stages_16_4<false, false>(re, im, false, tw, lt, nt); break;
            default: radix4_stage<false, false, 1>(re, im, false, tw, lt, nt); break;
        }
    } else {
        switch (step) {
            case 0: radix4_stage<true, false, 1>(re, im, true, tw, lt, nt); break;
            case 1: radix4_stages_16_4<true, false>(re, im, true, tw, lt, nt); break;
            case 2: swap_rows(re, im, lt, nt); break;
            case 3: radix4_stages_16_4<false, true>(re, im, true, tw, lt, nt); break;
            default: radix4_stage<false, true, 1>(re, im, true, tw, lt, nt); break;
        }
    }
}
__device__ void forward2(float* re, float* im, const float* tw) {
    for (int step = 0; step < kForwardSteps; ++step) {
        transform_step(false, step, re, im, tw, threadIdx.x, kThreads);
        __syncthreads();
    }
}
__device__ void inverse2(float* re, float* im, const float* tw) {
    for (int step = 0; step < kInverseSteps; ++step) {
        transform_step(true, step, re, im, tw, threadIdx.x, kThreads);
        __syncthreads();
    }
}

// Each thread keeps the 8 elements t + 512 k (k < 8) of the patch arrays in
// registers: row (t >> 6) + 8 k, column t & 63. Global scratch is written
// once and read once a patch (stores do not fill L1, so a store-then-load is
// an L2 round trip); the 32-lane sums read operands staged in the transform's
// shared buffer, which is idle until the forward transform.
constexpr int kPer = kArea / kThreads;  // 8
constexpr int kRowStep = kThreads / 64;
static_assert(kThreads % 64 == 0 && kPer * kThreads == kArea, "element ownership");

template <int C>
__global__ void __launch_bounds__(kThreads) da3d_tile_kernel(TileParams p) {
    extern __shared__ float shared[];
    float* are = shared;
    float* aim = are + kShared;
    float* bre = aim + kShared;  // RGB: channels 0 and 1 combined; staging for the sums
    float* bim = bre + kShared;
    // Staging for the sums: two arrays of kArea in the transform's buffer.
    float* st0 = shared;
    // with the second buffer, up to four arrays staged at once
    float* const stage[4] = {shared, shared + kArea, shared + 2 * kShared, shared + 2 * kShared + kArea};
    // Pyramid levels from kSharedLevel up live in shared memory.
    __shared__ float swm[kSharedLevelFloats];
    __shared__ float sums[16];
    __shared__ float centers[3];
    __shared__ int sh_pr, sh_pc, sh_go;
    __shared__ int lv_rows[32], lv_cols[32];
    __shared__ int lv_off[32];
    __shared__ int levels;
    __shared__ float tw[128];

    const int tile = blockIdx.x;
    const int tr = tile / p.cols_t, tc = tile % p.cols_t;
    const int r0 = tile_start(p.height, p.rows_t, tr), r1 = tile_start(p.height, p.rows_t, tr + 1);
    const int c0 = tile_start(p.width, p.cols_t, tc), c1 = tile_start(p.width, p.cols_t, tc + 1);
    const int core_h = r1 - r0, core_w = c1 - c0;
    const int ph = core_h + kS - 1, pw = core_w + kS - 1;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int ts = p.tile_w;
    const int my_col = threadIdx.x & 63, my_row = threadIdx.x >> 6;  // element k: row my_row + kRowStep k
    constexpr int kCenter = kR * kS + kR;
    constexpr int kCenterOwner = kCenter % kThreads, kCenterK = kCenter / kThreads;
    float* tnum = p.tile_num + static_cast<std::size_t>(tile) * C * p.tile_area;
    float* tden = p.tile_den + static_cast<std::size_t>(tile) * p.tile_area;
    float* wm = p.wm + static_cast<std::size_t>(tile) * p.wm_stride;
    for (int i = threadIdx.x; i < 128; i += kThreads) tw[i] = p.twiddle[i];

    for (int i = threadIdx.x; i < ph * pw; i += kThreads) {
        const std::size_t o = static_cast<std::size_t>(i / pw) * ts + i % pw;
        for (int c = 0; c < C; ++c) tnum[c * p.tile_area + o] = 0.f;
        tden[o] = 0.f;
    }
    if (threadIdx.x == 0) {
        levels = max(dev_bits(core_h - 1) + 1, dev_bits(core_w - 1) + 1);
        int rr = dev_next_pow2(core_h), cr = dev_next_pow2(core_w);
        int off = 0, soff = 0;
        for (int l = 0; l < levels; ++l) {
            lv_rows[l] = rr;
            lv_cols[l] = cr;
            if (l >= kSharedLevel) {
                lv_off[l] = soff;
                soff += (rr * cr + 3) & ~3;
            } else {
                lv_off[l] = off;
            }
            off += (rr * cr + 15) & ~15;
            rr = ((rr + 3) >> 2) << 1;
            cr = ((cr + 3) >> 2) << 1;
        }
    }
    __syncthreads();
    auto level = [&](int l) -> float* { return l >= kSharedLevel ? swm + lv_off[l] : wm + lv_off[l]; };
    {
        int vr = core_h, vc = core_w;
        const float inf = __int_as_float(0x7f800000);
        for (int l = 0; l < levels; ++l) {
            float* lvl = level(l);
            for (int i = threadIdx.x; i < lv_rows[l] * lv_cols[l]; i += kThreads) {
                const int r = i / lv_cols[l], c = i % lv_cols[l];
                lvl[i] = (r < vr && c < vc) ? 0.f : inf;
            }
            vr = (vr + 1) >> 1;
            vc = (vc + 1) >> 1;
        }
    }
    __syncthreads();

    const float sigma2 = p.sigma2;
    const float gamma_r_sigma2 = nss::kDa3dGammaR * sigma2;
    const float inv_reg = 1.f / (gamma_r_sigma2 * nss::kDa3dRegressionRangeScale);
    const float inv_k = 1.f / gamma_r_sigma2;
    const float cc = static_cast<float>(my_col - kR);
    unsigned long long patches = 0, guide_only = 0;
    float y[C][kPer], g[C][kPer], kv[kPer];
#ifdef NSS_DA3D_PHASES  // profiling aid: cycles per phase, printed per tile
    long long phs[8] = {}, last = clock64();
#define PH(n) do { if (threadIdx.x == 0) { const long long now = clock64(); phs[n] += now - last; last = now; } } while (0)
#else
#define PH(n) do {} while (0)
#endif

    for (;;) {
        if (threadIdx.x == 0) {
            const float* top = level(levels - 1);
            sh_go = top[0] < nss::kDa3dThreshold;
            int r = 0, c = 0;
            for (int l = levels - 2; l >= 0; --l) {
                r <<= 1;
                c <<= 1;
                const float* lvl = level(l);
                const int lc = lv_cols[l];
                const float v00 = lvl[r * lc + c], v10 = lvl[(r + 1) * lc + c];
                const float v01 = lvl[r * lc + c + 1], v11 = lvl[(r + 1) * lc + c + 1];
                int br = r, bc = c;
                float best = v00;
                if (v10 < best) { best = v10; br = r + 1; bc = c; }
                if (v01 < best) { best = v01; br = r; bc = c + 1; }
                if (v11 < best) { best = v11; br = r + 1; bc = c + 1; }
                r = br;
                c = bc;
            }
            sh_pr = r;
            sh_pc = c;
        }
        __syncthreads();
        if (!sh_go) break;
        PH(0);
        const int pr = sh_pr, pc = sh_pc;
        ++patches;

        // The patch into registers (the plane mirrored).
        {
            const int sx = symmetric(c0 - kR + pc + my_col, p.width);
#pragma unroll
            for (int k = 0; k < kPer; ++k) {
                const int sy = symmetric(r0 - kR + pr + my_row + kRowStep * k, p.height);
                const std::size_t o = static_cast<std::size_t>(sy) * p.width + sx;
#pragma unroll
                for (int c = 0; c < C; ++c) {
                    y[c][k] = p.plane_noisy[c][o];
                    g[c][k] = p.plane_guide[c][o];
                }
            }
        }
        if (threadIdx.x == kCenterOwner)
            for (int c = 0; c < C; ++c) centers[c] = g[c][kCenterK];
        __syncthreads();
        float center[C];
#pragma unroll
        for (int c = 0; c < C; ++c) center[c] = centers[c];
        // k_reg (in kv), staged with y of channel 0
#pragma unroll
        for (int k = 0; k < kPer; ++k) {
            const int i = threadIdx.x + kThreads * k;
            float e = g[0][k] - center[0];
            float x = e * e;
#pragma unroll
            for (int c = 1; c < C; ++c) {
                e = g[c][k] - center[c];
                x = fmaf(e, e, x);
            }
            x = fmaf(x, inv_reg, p.dist_reg[i]);
            kv[k] = fastexp(-x);
            st0[i] = kv[k];
#pragma unroll
            for (int c = 0; c < C; ++c) stage[1 + c][i] = y[c][k];
        }
        __syncthreads();
        // warps 0-2: a, b, c; warps 3 + 2c, 4 + 2c: d and e of channel c (its y in stage 1 + c)
        if (warp < 3 + 2 * C) {
            float v;
            if (warp == 0) {
                v = warp_lane_sum([&](float acc, int i) {
                    const int dr = (i >> 6) - kR;
                    return fmaf(static_cast<float>(dr * dr), st0[i], acc);
                });
            } else if (warp == 1) {
                v = warp_lane_sum([&](float acc, int i) {
                    const int dr = (i >> 6) - kR, dc = (i & 63) - kR;
                    return fmaf(static_cast<float>(dr * dc), st0[i], acc);
                });
            } else if (warp == 2) {
                v = warp_lane_sum([&](float acc, int i) {
                    const int dc = (i & 63) - kR;
                    return fmaf(static_cast<float>(dc * dc), st0[i], acc);
                });
            } else {
                const int c = (warp - 3) / 2;
                const float central = center[c];
                const float* yc = stage[1 + c];
                if ((warp - 3) % 2 == 0) {
                    v = warp_lane_sum([&](float acc, int i) {
                        const float t = static_cast<float>((i >> 6) - kR) * (yc[i] - central);
                        return fmaf(t, st0[i], acc);
                    });
                } else {
                    v = warp_lane_sum([&](float acc, int i) {
                        const float t = static_cast<float>((i & 63) - kR) * (yc[i] - central);
                        return fmaf(t, st0[i], acc);
                    });
                }
            }
            if (lane == 0) sums[warp] = v;  // a, b, c, then d0, e0, d1, e1, ...
        }
        __syncthreads();
        PH(1);
        float slope_r[C], slope_c[C];
        {
            const float a = sums[0], b = sums[1], cq = sums[2];
            const float det = a * cq - b * b;
            const bool fit = !(fabsf(det) < nss::kDa3dRegressionEpsilon);
#pragma unroll
            for (int c = 0; c < C; ++c) {
                slope_r[c] = 0.f;
                slope_c[c] = 0.f;
                if (fit) {
                    const float dd = sums[3 + 2 * c], ee = sums[4 + 2 * c];
                    slope_r[c] = (cq * dd - b * ee) / det;
                    slope_c[c] = (a * ee - b * dd) / det;
                }
            }
        }
        // the plane subtracted from y and g
#pragma unroll
        for (int k = 0; k < kPer; ++k) {
            const float rr = static_cast<float>(my_row + kRowStep * k - kR);
#pragma unroll
            for (int c = 0; c < C; ++c) {
                const float v = fmaf(slope_c[c], cc, slope_r[c] * rr);
                y[c][k] = y[c][k] - v;
                g[c][k] = g[c][k] - v;
            }
        }
        if (threadIdx.x == kCenterOwner)
            for (int c = 0; c < C; ++c) centers[c] = g[c][kCenterK];
        __syncthreads();
#pragma unroll
        for (int c = 0; c < C; ++c) center[c] = centers[c];
        // k, staged for mass and energy
#pragma unroll
        for (int k = 0; k < kPer; ++k) {
            const int i = threadIdx.x + kThreads * k;
            float e = g[0][k] - center[0];
            float x = e * e;
#pragma unroll
            for (int c = 1; c < C; ++c) {
                e = g[c][k] - center[c];
                x = fmaf(e, e, x);
            }
            x = fmaf(x, inv_k, p.dist_k[i]);
            kv[k] = fastexp(-x);
            st0[i] = kv[k];
        }
        __syncthreads();
        if (warp < 2) {
            const float v = warp == 0 ? warp_lane_sum([&](float acc, int i) { return acc + st0[i]; })
                                      : warp_lane_sum([&](float acc, int i) { return fmaf(st0[i], st0[i], acc); });
            if (lane == 0) sums[8 + warp] = v;
        }
        __syncthreads();
        PH(2);
        const float mass = sums[8], energy = sums[9];

        // Tile sums of this patch: loads first, then the stores.
        auto add_tile = [&](auto value_of) {
            float old[C][kPer], dold[kPer];
#pragma unroll
            for (int k = 0; k < kPer; ++k) {
                const std::size_t o = static_cast<std::size_t>(pr + my_row + kRowStep * k) * ts + pc + my_col;
#pragma unroll
                for (int c = 0; c < C; ++c) old[c][k] = tnum[c * p.tile_area + o];
                dold[k] = tden[o];
            }
#pragma unroll
            for (int k = 0; k < kPer; ++k) {
                const std::size_t o = static_cast<std::size_t>(pr + my_row + kRowStep * k) * ts + pc + my_col;
                const float kk = kv[k] * kv[k];
#pragma unroll
                for (int c = 0; c < C; ++c) tnum[c * p.tile_area + o] = value_of(c, k, old[c][k]);
                tden[o] = dold[k] + kk;
                kv[k] = kk;
            }
        };

        if (mass < nss::kDa3dGuideOnlyMass) {
            ++guide_only;
            add_tile([&](int c, int k, float old) {
                const float rr = static_cast<float>(my_row + kRowStep * k - kR);
                const float pl = fmaf(slope_c[c], cc, slope_r[c] * rr);
                const float kk = kv[k] * kv[k];
                return fmaf(g[c][k] + pl, kk, old);
            });
        } else {
            // averages: k in stage 0, y (then g) of each channel in the others;
            // one round when they all fit
            float avg[C], gavg[C];
            constexpr int kRounds = 2 * C + 1 <= 4 ? 1 : 2;
            for (int round = 0; round < kRounds; ++round) {
#pragma unroll
                for (int k = 0; k < kPer; ++k)
#pragma unroll
                    for (int c = 0; c < C; ++c) {
                        if (kRounds == 1) {
                            stage[1 + c][threadIdx.x + kThreads * k] = y[c][k];
                            stage[1 + C + c][threadIdx.x + kThreads * k] = g[c][k];
                        } else {
                            stage[1 + c][threadIdx.x + kThreads * k] = round ? g[c][k] : y[c][k];
                        }
                    }
                __syncthreads();
                const int chains = kRounds == 1 ? 2 * C : C;
                if (warp < chains) {
                    const float* src = stage[1 + warp];
                    const float v = warp_lane_sum([&](float acc, int i) { return fmaf(st0[i], src[i], acc); });
                    // stage 1 + c: y of channel c (or g in round 1); with one round, g at 1 + C + c
                    const int c = warp % C;
                    const int which = kRounds == 1 ? warp / C : round;
                    if (lane == 0) sums[10 + 2 * c + which] = v;
                }
                __syncthreads();
            }
            PH(3);
#pragma unroll
            for (int c = 0; c < C; ++c) {
                avg[c] = sums[10 + 2 * c] / mass;
                gavg[c] = sums[11 + 2 * c] / mass;
            }
            const float coef = -nss::kDa3dGammaF * (energy * sigma2);
            const float inv_n = 1.f / static_cast<float>(kArea);
            // Forward transform of channel c into a, then Y' = K Y in place:
            // the thread of the lower index of a pair (f, -f) writes both (the
            // CPU's value at -f is the conjugate of its value at f, bit for bit).
            // y_m + i g_m of channel c into (re, im)
            auto prep = [&](int c, float* re_, float* im_) {
#pragma unroll
                for (int k = 0; k < kPer; ++k) {
                    const float rest = 1.f - kv[k];
                    const int si = (my_row + kRowStep * k) * kPitch + my_col;
                    re_[si] = fmaf(kv[k], y[c][k], rest * avg[c]);
                    im_[si] = fmaf(kv[k], g[c][k], rest * gavg[c]);
                }
            };
            // Y' = K Y in place: the thread of the lower index of a pair (f, -f)
            // writes both (the CPU's value at -f is the conjugate of its value
            // at f, bit for bit)
            auto shrink = [&](float* re_, float* im_, int lt, int nt) {
                for (int i = lt; i < kArea; i += nt) {
                    const int q = negrow(i >> 6) * 64 + negrow(i & 63);
                    if (q < i) continue;
                    const int si = (i >> 6) * kPitch + (i & 63), sq = (q >> 6) * kPitch + (q & 63);
                    const float zr = re_[si], zi = im_[si], qr = re_[sq], qi = im_[sq];
                    const float yr = 0.5f * (zr + qr);
                    const float yi = 0.5f * (zi - qi);
                    const float gr = 0.5f * (zi + qi);
                    const float gi = 0.5f * (qr - zr);
                    float K = fastexp(coef / fmaf(gr, gr, gi * gi));
                    if (i == 0) K = 1.f;
                    const float ry = K * yr, iy = K * yi;
                    re_[si] = ry;
                    im_[si] = iy;
                    if (q != i) {
                        re_[sq] = ry;
                        im_[sq] = -iy;
                    }
                }
            };
            float result[C][kPer];
            auto take = [&](int c, const float* yout) {
#pragma unroll
                for (int k = 0; k < kPer; ++k) {
                    const int row = my_row + kRowStep * k;
                    const float rr = static_cast<float>(row - kR);
                    const float pl = fmaf(slope_c[c], cc, slope_r[c] * rr);
                    const float rest = 1.f - kv[k];
                    float v = yout[rev4(row) * kPitch + my_col] * inv_n - rest * avg[c];
                    result[c][k] = fmaf(pl, kv[k], v);
                }
            };
            if (C == 1) {
                prep(0, are, aim);
                __syncthreads();
                forward2(are, aim, tw);
                shrink(are, aim, threadIdx.x, kThreads);
                __syncthreads();
                inverse2(are, aim, tw);
                take(0, are);
            } else {
                // two transforms at a time, one on each half of the block
                const bool lower = threadIdx.x < kThreads / 2;
                const int lt = threadIdx.x % (kThreads / 2), nt = kThreads / 2;
                float* hre = lower ? are : bre;
                float* him = lower ? aim : bim;
                prep(0, are, aim);
                prep(1, bre, bim);
                __syncthreads();
                for (int step = 0; step < kForwardSteps; ++step) {
                    transform_step(false, step, hre, him, tw, lt, nt);
                    __syncthreads();
                }
                shrink(hre, him, lt, nt);
                __syncthreads();
                // W0 = Y'0 + i Y'1 in a
                for (int i = threadIdx.x; i < kArea; i += kThreads) {
                    const int si = (i >> 6) * kPitch + (i & 63);
                    are[si] = are[si] - bim[si];
                    aim[si] = aim[si] + bre[si];
                }
                __syncthreads();
                prep(2, bre, bim);
                __syncthreads();
                // the inverse of W0 (lower half) beside channel 2's forward (upper)
                for (int step = 0; step < kInverseSteps; ++step) {
                    if (lower) transform_step(true, step, are, aim, tw, lt, nt);
                    else if (step < kForwardSteps) transform_step(false, step, bre, bim, tw, lt, nt);
                    __syncthreads();
                }
                take(0, are);
                take(1, aim);
                shrink(bre, bim, threadIdx.x, kThreads);
                __syncthreads();
                inverse2(bre, bim, tw);
                take(2, bre);
            }
            add_tile([&](int c, int k, float old) { return fmaf(result[c][k], kv[k], old); });
        }
        __syncthreads();
        PH(5);

        // IncreaseWeights(k^2, pr - r, pc - r): level 0 from the registers
        const int row0 = pr - kR, col0 = pc - kR;
        const int first_r = max(0, row0), last_r = min(core_h, row0 + kS) - 1;
        const int first_c = max(0, col0), last_c = min(core_w, col0 + kS) - 1;
        {
            float* lvl = level(0);
            const int c = col0 + my_col;
            if (c >= first_c && c <= last_c) {
                float old[kPer];
#pragma unroll
                for (int k = 0; k < kPer; ++k) {
                    const int r = row0 + my_row + kRowStep * k;
                    old[k] = (r >= first_r && r <= last_r) ? lvl[r * lv_cols[0] + c] : 0.f;
                }
#pragma unroll
                for (int k = 0; k < kPer; ++k) {
                    const int r = row0 + my_row + kRowStep * k;
                    if (r >= first_r && r <= last_r) lvl[r * lv_cols[0] + c] = old[k] + kv[k];
                }
            }
            __syncthreads();
        }
        for (int l = 1; l < levels; ++l) {
            const int fr = first_r >> l, lr = last_r >> l, fc = first_c >> l, lc_ = last_c >> l;
            const int nr = lr - fr + 1, nc = lc_ - fc + 1;
            const float* lo = level(l - 1);
            float* lvl = level(l);
            const int loc = lv_cols[l - 1];
            for (int i = threadIdx.x; i < nr * nc; i += kThreads) {
                const int r = fr + i / nc, c = fc + i % nc;
                const float* q = lo + (2 * r) * loc + 2 * c;
                lvl[r * lv_cols[l] + c] = fminf(fminf(q[0], q[1]), fminf(q[loc], q[loc + 1]));
            }
            __syncthreads();
        }
        PH(6);
    }
#ifdef NSS_DA3D_PHASES
    if (threadIdx.x == 0)
        printf("PH %d %llu %llu %lld %lld %lld %lld %lld %lld %lld\n", blockIdx.x, patches, guide_only, phs[0], phs[1],
               phs[2], phs[3], phs[4], phs[5], phs[6]);
#endif
    if (threadIdx.x == 0) {
        atomicAdd(p.stats, patches);
        atomicAdd(p.stats + 1, guide_only);
    }
}

struct MergeParams {
    int width, height, rows_t, cols_t, tile_w;
    std::size_t tile_area;
    const float* tile_num;
    const float* tile_den;
    float* out[3];
    bool rgb;
    float rs2, rs3, rs6;
};

// The tiles' sums in tile order, divided, and (RGB) transformed back.
template <int C>
__global__ void da3d_merge_kernel(MergeParams p) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int yy = blockIdx.y;
    if (x >= p.width) return;
    float num[3] = {0.f, 0.f, 0.f}, den = 0.f;
    for (int tr = 0; tr < p.rows_t; ++tr) {
        const int r0 = tile_start(p.height, p.rows_t, tr), r1 = tile_start(p.height, p.rows_t, tr + 1);
        if (yy < r0 - kR || yy >= r1 + kAfter) continue;
        for (int tc = 0; tc < p.cols_t; ++tc) {
            const int c0 = tile_start(p.width, p.cols_t, tc), c1 = tile_start(p.width, p.cols_t, tc + 1);
            if (x < c0 - kR || x >= c1 + kAfter) continue;
            const int tile = tr * p.cols_t + tc;
            const std::size_t o = static_cast<std::size_t>(yy - (r0 - kR)) * p.tile_w + (x - (c0 - kR));
            for (int c = 0; c < C; ++c) num[c] = num[c] + p.tile_num[(static_cast<std::size_t>(tile) * C + c) * p.tile_area + o];
            den = den + p.tile_den[static_cast<std::size_t>(tile) * p.tile_area + o];
        }
    }
    float v[3];
    for (int c = 0; c < C; ++c) v[c] = num[c] / den;
    if (C == 3 && p.rgb) {
        const float a = v[0], b = v[1], w = v[2];
        v[0] = (p.rs2 * a + p.rs3 * b + w) / p.rs6;
        v[1] = (a - p.rs2 * w) / p.rs3;
        v[2] = (p.rs2 * a - p.rs3 * b + w) / p.rs6;
    }
    const std::size_t o = static_cast<std::size_t>(yy) * p.width + x;
    for (int c = 0; c < C; ++c) p.out[c][o] = v[c];
}

__global__ void da3d_color_kernel(const float* r, const float* g, const float* b, float* out, std::size_t n, float rs2,
                                  float rs3, float rs6) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float vr = r[i], vg = g[i], vb = b[i];
    out[i] = (vr + vg + vb) / rs3;
    out[n + i] = (vr - vb) / rs2;
    out[2 * n + i] = (vr - 2 * vg + vb) / rs6;
}

struct Layout {
    int rows_t, cols_t, tiles, tile_w, tile_h;
    std::size_t tile_area, wm_stride, num, den, wm, scratch, total;
};

Layout layout(int width, int height, int channels) {
    Layout l{};
    l.rows_t = nss::da3d_tile_count(height);
    l.cols_t = nss::da3d_tile_count(width);
    l.tiles = l.rows_t * l.cols_t;
    l.tile_w = nss::da3d_tile_max(width) + kS - 1;
    l.tile_h = nss::da3d_tile_max(height) + kS - 1;
    l.tile_area = static_cast<std::size_t>(l.tile_w) * l.tile_h;
    l.wm_stride = align16(wm_floats(nss::da3d_tile_max(height), nss::da3d_tile_max(width)));
    const auto bytes = [](std::size_t floats) { return (floats * sizeof(float) + 255) & ~std::size_t{255}; };
    l.num = bytes(static_cast<std::size_t>(l.tiles) * channels * l.tile_area);
    l.den = bytes(static_cast<std::size_t>(l.tiles) * l.tile_area);
    l.wm = bytes(static_cast<std::size_t>(l.tiles) * l.wm_stride);
    l.scratch = 0;
    l.total = l.num + l.den + l.wm + l.scratch;
    return l;
}

}  // namespace

std::size_t da3d_work_bytes(int width, int height, int channels) { return layout(width, height, channels).total; }

std::size_t da3d_table_floats() { return 128 + 2 * kArea; }

void da3d_fill_tables(float* host) {
    const double pi = 3.14159265358979323846;
    for (int t = 0; t < 64; ++t) {
        host[t] = static_cast<float>(std::cos(2.0 * pi * t / 64.0));
        host[64 + t] = static_cast<float>(-std::sin(2.0 * pi * t / 64.0));
    }
    const float sigma_s2 = nss::kDa3dSigmaS * nss::kDa3dSigmaS;
    const float sigma_sr2 = sigma_s2 * nss::kDa3dRegressionSpatialScale;
    for (int row = 0; row < kS; ++row) {
        for (int col = 0; col < kS; ++col) {
            const int dr = row - kR, dc = col - kR;
            host[128 + row * kS + col] = static_cast<float>(dr * dr + dc * dc) / (2 * sigma_sr2);
            host[128 + kArea + row * kS + col] = static_cast<float>(dr * dr + dc * dc) / (2 * sigma_s2);
        }
    }
}

void da3d_run(const Da3dUnit& u, const Da3dTables& tables, void* work, std::size_t work_bytes,
              unsigned long long* stats_device, cudaStream_t stream) {
    const int C = u.channels;
    const Layout l = layout(u.width, u.height, C);
    if (work_bytes < l.total) throw std::invalid_argument("nss_cuda: DA3D work buffer too small");
    auto* base = static_cast<std::uint8_t*>(work);
    const std::size_t n = static_cast<std::size_t>(u.width) * u.height;
    const float rs2 = std::sqrt(2.f), rs3 = std::sqrt(3.f), rs6 = std::sqrt(6.f);
    TileParams p{};
    if (u.rgb) {
        da3d_color_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(u.noisy[0], u.noisy[1], u.noisy[2],
                                                                                    u.transformed, n, rs2, rs3, rs6);
        NSS_CUDA_CHECK_LAUNCH();
        da3d_color_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(u.guide[0], u.guide[1], u.guide[2],
                                                                                    u.transformed + 3 * n, n, rs2, rs3, rs6);
        NSS_CUDA_CHECK_LAUNCH();
        for (int c = 0; c < 3; ++c) {
            p.plane_noisy[c] = u.transformed + c * n;
            p.plane_guide[c] = u.transformed + (3 + c) * n;
        }
    } else {
        for (int c = 0; c < C; ++c) {
            p.plane_noisy[c] = u.noisy[c];
            p.plane_guide[c] = u.guide[c];
        }
    }
    p.width = u.width;
    p.height = u.height;
    p.rows_t = l.rows_t;
    p.cols_t = l.cols_t;
    p.tile_w = l.tile_w;
    p.tile_area = l.tile_area;
    p.wm_stride = l.wm_stride;
    p.sigma2 = u.sigma * u.sigma;
    p.twiddle = tables.twiddle;
    p.dist_reg = tables.dist;
    p.dist_k = tables.dist + kArea;
    p.tile_num = reinterpret_cast<float*>(base);
    p.tile_den = reinterpret_cast<float*>(base + l.num);
    p.wm = reinterpret_cast<float*>(base + l.num + l.den);
    p.scratch = reinterpret_cast<float*>(base + l.num + l.den + l.wm);
    p.stats = stats_device;
    NSS_CUDA_CHECK(cudaMemsetAsync(stats_device, 0, 2 * sizeof(unsigned long long), stream));
    const std::size_t smem = 4 * kShared * sizeof(float);  // the second buffer also stages sums
    if (C == 3) {
        NSS_CUDA_CHECK(cudaFuncSetAttribute(da3d_tile_kernel<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
        da3d_tile_kernel<3><<<l.tiles, kThreads, smem, stream>>>(p);
    } else {
        NSS_CUDA_CHECK(cudaFuncSetAttribute(da3d_tile_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
        da3d_tile_kernel<1><<<l.tiles, kThreads, smem, stream>>>(p);
    }
    NSS_CUDA_CHECK_LAUNCH();
    MergeParams m{};
    m.width = u.width;
    m.height = u.height;
    m.rows_t = l.rows_t;
    m.cols_t = l.cols_t;
    m.tile_w = l.tile_w;
    m.tile_area = l.tile_area;
    m.tile_num = p.tile_num;
    m.tile_den = p.tile_den;
    for (int c = 0; c < C; ++c) m.out[c] = u.out[c];
    m.rgb = u.rgb;
    m.rs2 = rs2;
    m.rs3 = rs3;
    m.rs6 = rs6;
    const dim3 grid((u.width + 127) / 128, u.height);
    if (C == 3) da3d_merge_kernel<3><<<grid, 128, 0, stream>>>(m);
    else da3d_merge_kernel<1><<<grid, 128, 0, stream>>>(m);
    NSS_CUDA_CHECK_LAUNCH();
}

}  // namespace nss_cuda
