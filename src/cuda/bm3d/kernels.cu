// SPDX-License-Identifier: GPL-2.0-only
#include "cuda/bm3d/kernels.hpp"
#include "cuda/bm3d/dct_codelets.cuh"
#include "cuda/common/fixed_accumulate.cuh"
#include "cuda/runtime/error.hpp"

#include <cstddef>
#include <stdexcept>

namespace nss_cuda {
namespace {

constexpr float kHardLambda = 2.7f;  // nss::kBmHardLambda

// 8-point DCT-II / DCT-III butterflies (FFTW e10_8 / e01_8, the arithmetic of
// the CPU's bm3d_filter8). Each is 4x the orthonormal transform, so the 3-D
// forward scales coefficients by 64 and the round trip by 4096: both exact
// powers of two, applied to the threshold and to the result.
constexpr float kDctP414 = 0.414213562373095048801688724209698078569671875f;
constexpr float kDctP1847 = 1.847759065022573512256366378793576573644833252f;
constexpr float kDctP198 = 0.198912367379658006911597622644676228597850501f;
constexpr float kDctP1961 = 1.961570560806460898252364472268478073947867462f;
constexpr float kDctP1414 = 1.414213562373095048801688724209698078569671875f;
constexpr float kDctP668 = 0.668178637919298919997757686523080761552472251f;
constexpr float kDctP1662 = 1.662939224605090474157576755235811513477121624f;
constexpr float kDctP707 = 0.707106781186547524400844362104849039284835938f;

__device__ __forceinline__ void dct8_forward(float* b, int s) {
    const float t1 = b[0], t2 = b[7 * s];
    const float t3 = t1 - t2, tj = t1 + t2;
    const float tc = b[4 * s], td = b[3 * s];
    const float te = tc - td, tk = tc + td;
    const float t4 = b[2 * s], t5 = b[5 * s];
    const float t6 = t4 - t5;
    const float t7 = b[s], t8 = b[6 * s];
    const float t9 = t7 - t8;
    const float ta = t6 + t9, tn = t7 + t8, tf = t6 - t9, tm = t4 + t5;
    const float tb = fmaf(-kDctP707, ta, t3);
    const float tg = fmaf(-kDctP707, tf, te);
    b[3 * s] = kDctP1662 * fmaf(kDctP668, tg, tb);
    b[5 * s] = -(kDctP1662 * fmaf(-kDctP668, tb, tg));
    const float tp = tj + tk, tq = tm + tn;
    b[4 * s] = kDctP1414 * (tp - tq);
    b[0] = kDctP1414 * (tp + tq);
    const float th = fmaf(kDctP707, ta, t3);
    const float ti = fmaf(kDctP707, tf, te);
    b[s] = kDctP1961 * fmaf(-kDctP198, ti, th);
    b[7 * s] = kDctP1961 * fmaf(kDctP198, th, ti);
    const float tl = tj - tk, to = tm - tn;
    b[2 * s] = kDctP1847 * fmaf(-kDctP414, to, tl);
    b[6 * s] = kDctP1847 * fmaf(kDctP414, tl, to);
}

__device__ __forceinline__ void dct8_inverse(float* b, int s) {
    const float t1 = kDctP1414 * b[0];
    const float t2 = b[4 * s];
    const float t3 = fmaf(kDctP1414, t2, t1);
    const float tj = fmaf(-kDctP1414, t2, t1);
    const float t4 = b[2 * s], t5 = b[6 * s];
    const float t6 = fmaf(kDctP414, t5, t4);
    const float tk = kDctP414 * t4 - t5;
    const float t8 = b[s], td = b[7 * s], t9 = b[5 * s], ta = b[3 * s];
    const float tb = t9 + ta, te = ta - t9;
    const float tc = fmaf(kDctP707, tb, t8);
    const float tn = fmaf(-kDctP707, te, td);
    const float tf = fmaf(kDctP707, te, td);
    const float tm = fmaf(-kDctP707, tb, t8);
    const float t7 = fmaf(kDctP1847, t6, t3);
    const float tg = fmaf(kDctP198, tf, tc);
    b[7 * s] = fmaf(-kDctP1961, tg, t7);
    b[0] = fmaf(kDctP1961, tg, t7);
    const float tp = fmaf(-kDctP1847, tk, tj);
    const float tq = fmaf(kDctP668, tm, tn);
    b[5 * s] = fmaf(-kDctP1662, tq, tp);
    b[2 * s] = fmaf(kDctP1662, tq, tp);
    const float th = fmaf(-kDctP1847, t6, t3);
    const float ti = fmaf(-kDctP198, tc, tf);
    b[3 * s] = fmaf(-kDctP1961, ti, th);
    b[4 * s] = fmaf(kDctP1961, ti, th);
    const float tl = fmaf(kDctP1847, tk, tj);
    const float to = fmaf(-kDctP668, tn, tm);
    b[6 * s] = fmaf(-kDctP1662, to, tl);
    b[s] = fmaf(kDctP1662, to, tl);
}

// --- Every other shape: one block per group, sizes known at compile time ----
//
// The cube lives in shared memory when it fits (it is written to `values`
// once at the end), the block has as many threads as a transform pass has
// lines, and the line transforms are butterflies wherever one exists.

// Orthonormal 2- and 4-point DCT-II / DCT-III as the matrix product the CPU
// uses for these sizes (same terms in the same order).
constexpr float kDct4A = 0.653281482438188263928322f;  // cos(pi / 8) / sqrt(2)
constexpr float kDct4B = 0.270598050073098492199862f;  // sin(pi / 8) / sqrt(2)

template <int N>
__device__ __forceinline__ constexpr float dct_small_term(int k, int j) {
    if constexpr (N == 2) {
        constexpr float t[2][2] = {{kDctP707, kDctP707}, {kDctP707, -kDctP707}};
        return t[k][j];
    } else {
        constexpr float t[4][4] = {{0.5f, 0.5f, 0.5f, 0.5f},
                                   {kDct4A, kDct4B, -kDct4B, -kDct4A},
                                   {0.5f, -0.5f, -0.5f, 0.5f},
                                   {kDct4B, -kDct4A, kDct4A, -kDct4B}};
        return t[k][j];
    }
}

template <int N>
__device__ __forceinline__ void dct_small(float* base, int stride, bool inverse) {
    float in[N];
#pragma unroll
    for (int j = 0; j < N; ++j) in[j] = base[j * stride];
#pragma unroll
    for (int k = 0; k < N; ++k) {
        float acc = 0.f;
#pragma unroll
        for (int j = 0; j < N; ++j) {
            acc = fmaf(inverse ? dct_small_term<N>(j, k) : dct_small_term<N>(k, j), in[j], acc);
        }
        base[k * stride] = acc;
    }
}

// Scale of dct_line_fixed<N> against the orthonormal transform.
template <int N>
constexpr float kLineScale = N == 8 ? 4.f : 1.f;

template <int N>
__device__ __forceinline__ void dct_line_fixed(float* base, int stride, bool inverse) {
    if constexpr (N == 1) {
        return;
    } else if constexpr (N == 8) {
        if (inverse) {
            dct8_inverse(base, stride);
        } else {
            dct8_forward(base, stride);
        }
    } else if constexpr (N == 12 || N == 16 || N == 32 || N == 64) {
        float in[8 * N], out[8 * N];
#pragma unroll
        for (int k = 0; k < N; ++k) in[8 * k] = base[k * stride];
        if constexpr (N == 12) {
            inverse ? codelet::nss_dct12_inv_is8(0, in, out) : codelet::nss_dct12_fwd_is8(0, in, out);
        } else if constexpr (N == 16) {
            inverse ? codelet::nss_dct16_inv_is8(0, in, out) : codelet::nss_dct16_fwd_is8(0, in, out);
        } else if constexpr (N == 32) {
            inverse ? codelet::nss_dct32_inv_is8(0, in, out) : codelet::nss_dct32_fwd_is8(0, in, out);
        } else {
            inverse ? codelet::nss_dct64_inv_is8(0, in, out) : codelet::nss_dct64_fwd_is8(0, in, out);
        }
#pragma unroll
        for (int k = 0; k < N; ++k) base[k * stride] = out[8 * k];
    } else {
        dct_small<N>(base, stride, inverse);  // 2 and 4
    }
}

// Sum over the block of one value per thread: lanes by shuffles, then the
// warps in order. Fixed shape for a fixed block size. blockDim is a multiple
// of 32.
template <class T>
__device__ __forceinline__ T block_total(T value, T* per_warp) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_xor_sync(0xffffffffu, value, offset);
    if ((threadIdx.x & 31) == 0) per_warp[threadIdx.x >> 5] = value;
    __syncthreads();
    T total = 0;
    for (int w = 0; w < static_cast<int>(blockDim.x >> 5); ++w) total += per_warp[w];
    __syncthreads();
    return total;
}

template <int B, int G>
__device__ __forceinline__ void transform_cube(float* cube, bool inverse) {
    constexpr int kArea = B * B;
    const int tid = threadIdx.x, threads = blockDim.x;
    if (inverse) {
        for (int f = tid; f < kArea; f += threads) dct_line_fixed<G>(cube + f, kArea, true);
        __syncthreads();
    }
    for (int i = tid; i < G * B; i += threads) dct_line_fixed<B>(cube + i * B, 1, inverse);
    __syncthreads();
    for (int i = tid; i < G * B; i += threads) dct_line_fixed<B>(cube + (i / B) * kArea + i % B, B, inverse);
    __syncthreads();
    if (!inverse) {
        for (int f = tid; f < kArea; f += threads) dct_line_fixed<G>(cube + f, kArea, false);
        __syncthreads();
    }
}

constexpr int kMaxWarps = 8;  // blocks of the shape kernels have at most 256 threads

template <int B, int G>
__global__ void __launch_bounds__(256) group_shape_kernel(Bm3dGroupArgs a, bool in_shared) {
    extern __shared__ float staged[];  // cube, then the reference cube; empty when !in_shared
    __shared__ float fsum[kMaxWarps];
    __shared__ int isum[kMaxWarps];
    constexpr int kArea = B * B, kCube = G * kArea;
    constexpr float kScale = kLineScale<B> * kLineScale<B> * kLineScale<G>;
    const int r = blockIdx.x, tid = threadIdx.x, threads = blockDim.x;
    float* const out = a.fused.num ? nullptr : a.values + static_cast<long long>(r) * kCube;
    float* const cube = in_shared ? staged : out;
    float* const refc = !a.ref ? nullptr : in_shared ? staged + kCube : a.ref_cube + static_cast<long long>(r) * kCube;
    const DeviceMatch* m = a.matches + static_cast<long long>(r) * G;
    const int kk = min(a.counts[r], G);

    for (int i = tid; i < kCube; i += threads) {
        const int g = i / kArea, p = i % kArea;
        float v = 0.f, rv = 0.f;
        if (g < kk) {
            const long long offset = static_cast<long long>(m[g].y + p / B) * a.pitch + m[g].x + p % B;
            v = a.src[m[g].t][offset];
            if (refc) rv = a.ref[m[g].t][offset];
        }
        cube[i] = v;
        if (refc) refc[i] = rv;
    }
    __syncthreads();
    transform_cube<B, G>(cube, false);
    if (refc) transform_cube<B, G>(refc, false);

    float weight;
    if (refc) {
        const float sig2 = kScale * kScale * a.sigma * a.sigma;
        float w2 = 0.f;
        for (int i = tid; i < kCube; i += threads) {
            float w = 1.f;
            if (i != 0) {
                const float q = refc[i] * refc[i];
                w = q / (q + sig2);
            }
            cube[i] *= w;
            w2 = fmaf(w, w, w2);
        }
        weight = 1.f / fmaxf(block_total(w2, fsum), 1e-12f);
    } else {
        const float thr = kScale * kHardLambda * a.sigma;
        int kept = 0;
        for (int i = tid; i < kCube; i += threads) {
            if (i != 0 && fabsf(cube[i]) < thr) {
                cube[i] = 0.f;
            } else {
                ++kept;
            }
        }
        weight = 1.f / static_cast<float>(max(block_total(kept, isum), 1));
    }
    transform_cube<B, G>(cube, true);
    constexpr float kUnscale = 1.f / (kScale * kScale);
    if (a.fused.num) {  // in_shared
        for (int i = tid; i < kCube; i += threads) {
            const int g = i / kArea, p = i % kArea;
            if (g >= kk) continue;
            const long long at = m[g].t * static_cast<long long>(a.fused.slice_step) +
                                 static_cast<long long>(m[g].y + p / B) * a.fused.pitch + m[g].x + p % B;
            fixed_add(a.fused.num + at, weight * (cube[i] * kUnscale));
            fixed_add(a.fused.den + at, weight);
        }
        return;
    }
    if (in_shared) {
        for (int i = tid; i < kCube; i += threads) out[i] = cube[i] * kUnscale;
    } else if (kUnscale != 1.f) {
        for (int i = tid; i < kCube; i += threads) out[i] *= kUnscale;
    }
    for (int g = tid; g < G; g += threads) {
        a.patches[static_cast<long long>(r) * G + g] =
            g < kk ? AggregatePatch{m[g].x, m[g].y, m[g].t, weight} : AggregatePatch{0, 0, -1, 0.f};
    }
}

constexpr std::size_t kCubeSharedBytes = 24 * 1024;  // per block; larger cubes stay in `values`

std::size_t cube_bytes(int block, int group, bool wiener) {
    return static_cast<std::size_t>(group) * block * block * sizeof(float) * (wiener ? 2 : 1);
}

template <int B, int G>
void launch_shape(const Bm3dGroupArgs& args, cudaStream_t stream) {
    const std::size_t bytes = cube_bytes(B, G, args.ref != nullptr);
    const bool in_shared = bytes <= kCubeSharedBytes;
    if (args.fused.num && !in_shared) throw std::logic_error("nss_cuda: BM3D shape cannot aggregate in the kernel");
    const int lines = G * B > B * B ? G * B : B * B;
    const int threads = lines >= 256 ? 256 : (lines + 31) / 32 * 32;
    group_shape_kernel<B, G><<<args.batch, threads, in_shared ? bytes : 0, stream>>>(args, in_shared);
}

template <int B>
bool launch_block(const Bm3dGroupArgs& args, cudaStream_t stream) {
    switch (args.group) {
    case 1: launch_shape<B, 1>(args, stream); return true;
    case 2: launch_shape<B, 2>(args, stream); return true;
    case 4: launch_shape<B, 4>(args, stream); return true;
    case 8: launch_shape<B, 8>(args, stream); return true;
    case 16: launch_shape<B, 16>(args, stream); return true;
    case 32: launch_shape<B, 32>(args, stream); return true;
    case 64: launch_shape<B, 64>(args, stream); return true;
    default: return false;
    }
}

bool launch_shapes(const Bm3dGroupArgs& args, cudaStream_t stream) {
    switch (args.block) {
    case 1: return launch_block<1>(args, stream);
    case 2: return launch_block<2>(args, stream);
    case 4: return launch_block<4>(args, stream);
    case 8: return launch_block<8>(args, stream);
    case 12: return launch_block<12>(args, stream);
    case 16: return launch_block<16>(args, stream);
    case 32: return launch_block<32>(args, stream);
    default: return false;
    }
}


// --- Shapes whose cube fits registers: one warp filters 32 / B groups -------
//
// Lane l serves group l / B with sub-lane j = l % B and holds G * B samples
// v[p * B + r]: patch p, row r, column j. The transforms along r and p run in
// registers; the one along the columns runs after a B x B transposition
// across the group's lanes through a small shared buffer.
constexpr int kWarpStride = 33;  // one more than the warp: conflict-free banks both ways

// Transposes, for every p, the B x B matrix whose columns are the lanes of a
// group: v[p * B + i] of sub-lane j <-> v[p * B + j] of sub-lane i.
template <int B, int G>
__device__ __forceinline__ void transpose_lanes(float* v, float* buffer, int lane) {
    const int sub = lane % B, base = lane - sub;
#pragma unroll
    for (int p = 0; p < G; ++p) {
#pragma unroll
        for (int i = 0; i < B; ++i) buffer[i * kWarpStride + lane] = v[p * B + i];
        __syncwarp();
#pragma unroll
        for (int i = 0; i < B; ++i) v[p * B + i] = buffer[sub * kWarpStride + base + i];
        __syncwarp();
    }
}

// Loads the lane's column of the group's patches from `planes` (zero for the
// slots beyond the group's count) and applies the forward transform. On
// return the lane is the row frequency and the index within a patch the
// column frequency: the DC coefficient is v[0] of sub-lane 0.
template <int B, int G>
__device__ __forceinline__ void forward_lanes(float* v, const float* const* planes, const DeviceMatch* m, int kk,
                                              int pitch, float* buffer, int lane) {
    const int sub = lane % B;
#pragma unroll
    for (int p = 0; p < G; ++p) {
        const DeviceMatch& mp = m[p < kk ? p : 0];
        const float* at = planes[mp.t] + static_cast<long long>(mp.y) * pitch + mp.x + sub;
#pragma unroll
        for (int row = 0; row < B; ++row) v[p * B + row] = p < kk ? at[row * pitch] : 0.f;
    }
#pragma unroll
    for (int p = 0; p < G; ++p) dct_line_fixed<B>(v + p * B, 1, false);  // down the lane's column
#pragma unroll
    for (int row = 0; row < B; ++row) dct_line_fixed<G>(v + row, B, false);  // group axis
    transpose_lanes<B, G>(v, buffer, lane);
#pragma unroll
    for (int p = 0; p < G; ++p) dct_line_fixed<B>(v + p * B, 1, false);  // along the row the lane now holds
}

// Sum over the B lanes of a group (B a power of two).
template <int B, class T>
__device__ __forceinline__ T group_total(T value) {
#pragma unroll
    for (int offset = 1; offset < B; offset <<= 1) value += __shfl_xor_sync(0xffffffffu, value, offset);
    return value;
}

// Fused: the kernel adds its weighted patches to a.fused instead of storing
// them. A separate instantiation, so that the storing kernel's register
// allocation is not disturbed.
template <int B, int G, bool Wiener, bool Fused>
__global__ void __launch_bounds__(32) group_warp_kernel(Bm3dGroupArgs a) {
    __shared__ float buffer[B * kWarpStride];
    constexpr int kLane = G * B, kCube = G * B * B, kPerWarp = 32 / B;
    constexpr float kScale = kLineScale<B> * kLineScale<B> * kLineScale<G>;
    const int lane = threadIdx.x, sub = lane % B;
    const int r = blockIdx.x * kPerWarp + lane / B;
    const bool live = r < a.batch;
    const int rr = live ? r : a.batch - 1;  // idle lanes mirror the last group and store nothing
    const DeviceMatch* m = a.matches + static_cast<long long>(rr) * G;
    const int kk = min(a.counts[rr], G);
    float v[kLane];
    float weight;
    if constexpr (Wiener) {
        // Gains from the reference cube, parked in shared memory while the
        // registers transform the noisy cube.
        __shared__ float gains[32 * kLane];
        float* gain = gains + lane * kLane;
        forward_lanes<B, G>(v, a.ref, m, kk, a.pitch, buffer, lane);
        const float sig2 = kScale * kScale * a.sigma * a.sigma;
        float w2 = 0.f;
#pragma unroll
        for (int i = 0; i < kLane; ++i) {
            const float q = v[i] * v[i];
            const float w = (i == 0 && sub == 0) ? 1.f : q / (q + sig2);
            gain[i] = w;
            w2 = fmaf(w, w, w2);
        }
        weight = 1.f / fmaxf(group_total<B>(w2), 1e-12f);
        forward_lanes<B, G>(v, a.src, m, kk, a.pitch, buffer, lane);
#pragma unroll
        for (int i = 0; i < kLane; ++i) v[i] *= gain[i];
    } else {
        forward_lanes<B, G>(v, a.src, m, kk, a.pitch, buffer, lane);
        const float thr = kScale * kHardLambda * a.sigma;
        int kept = 0;
#pragma unroll
        for (int i = 0; i < kLane; ++i) {
            const bool keep = (i == 0 && sub == 0) || !(fabsf(v[i]) < thr);
            v[i] = keep ? v[i] : 0.f;
            kept += keep;
        }
        weight = 1.f / static_cast<float>(max(group_total<B>(kept), 1));
    }
#pragma unroll
    for (int p = 0; p < G; ++p) dct_line_fixed<B>(v + p * B, 1, true);
    transpose_lanes<B, G>(v, buffer, lane);
#pragma unroll
    for (int row = 0; row < B; ++row) dct_line_fixed<G>(v + row, B, true);
#pragma unroll
    for (int p = 0; p < G; ++p) dct_line_fixed<B>(v + p * B, 1, true);
    if (!live) return;
    constexpr float kUnscale = 1.f / (kScale * kScale);
    if constexpr (Fused) {
#pragma unroll
        for (int p = 0; p < G; ++p) {
            if (p >= kk) continue;
            const long long at = m[p].t * static_cast<long long>(a.fused.slice_step) +
                                 static_cast<long long>(m[p].y) * a.fused.pitch + m[p].x + sub;
#pragma unroll
            for (int row = 0; row < B; ++row) {
                fixed_add(a.fused.num + at + row * a.fused.pitch, weight * (v[p * B + row] * kUnscale));
                fixed_add(a.fused.den + at + row * a.fused.pitch, weight);
            }
        }
        return;
    }
    float* cube = a.values + static_cast<long long>(r) * kCube + sub;
#pragma unroll
    for (int p = 0; p < G; ++p) {
#pragma unroll
        for (int row = 0; row < B; ++row) cube[p * B * B + row * B] = v[p * B + row] * kUnscale;
    }
    for (int g = sub; g < G; g += B) {
        a.patches[static_cast<long long>(r) * G + g] =
            g < kk ? AggregatePatch{m[g].x, m[g].y, m[g].t, weight} : AggregatePatch{0, 0, -1, 0.f};
    }
}

template <int B, int G, bool Wiener>
void launch_warp_stage(const Bm3dGroupArgs& args, cudaStream_t stream) {
    const unsigned blocks = static_cast<unsigned>((args.batch + 32 / B - 1) / (32 / B));
    if (args.fused.num) {
        group_warp_kernel<B, G, Wiener, true><<<blocks, 32, 0, stream>>>(args);
    } else {
        group_warp_kernel<B, G, Wiener, false><<<blocks, 32, 0, stream>>>(args);
    }
}

template <int B, int G>
void launch_warp(const Bm3dGroupArgs& args, cudaStream_t stream) {
    if (args.ref) {
        launch_warp_stage<B, G, true>(args, stream);
    } else {
        launch_warp_stage<B, G, false>(args, stream);
    }
}

template <int B, int G>
void launch_warp_hard(const Bm3dGroupArgs& args, cudaStream_t stream) {
    launch_warp_stage<B, G, false>(args, stream);
}

// The shapes the warp kernel serves. Each was admitted on a paired
// measurement against the block kernel. Up to 128 samples per lane it wins
// for both stages. At 256 the kernel uses every register and spills a little:
// the hard-threshold stage still wins, the Wiener stage only for 16 / 16.
// At 512 samples per lane (8 / 64) it loses, except 16 / 32.
constexpr bool warp_serves(int block, int group, bool wiener) {
    switch (block * 100 + group) {
    case 402: case 404: case 408: case 416: case 432:
    case 802: case 804: case 808: case 816:
    case 1602: case 1604: case 1608: case 1616:
        return true;
    case 464: case 832: case 1632:
        return !wiener;
    default:
        return false;
    }
}

bool launch_warps(const Bm3dGroupArgs& args, cudaStream_t stream) {
    if (!warp_serves(args.block, args.group, args.ref != nullptr)) return false;
    switch (args.block * 100 + args.group) {
#define NSS_WARP(B, G) case B * 100 + G: launch_warp<B, G>(args, stream); return true;
#define NSS_WARP_HARD(B, G) case B * 100 + G: launch_warp_hard<B, G>(args, stream); return true;
    NSS_WARP(4, 2) NSS_WARP(4, 4) NSS_WARP(4, 8) NSS_WARP(4, 16) NSS_WARP(4, 32)
    NSS_WARP(8, 2) NSS_WARP(8, 4) NSS_WARP(8, 8) NSS_WARP(8, 16)
    NSS_WARP(16, 2) NSS_WARP(16, 4) NSS_WARP(16, 8) NSS_WARP(16, 16)
    NSS_WARP_HARD(4, 64) NSS_WARP_HARD(8, 32) NSS_WARP_HARD(16, 32)
#undef NSS_WARP
#undef NSS_WARP_HARD
    default: return false;
    }
}

}  // namespace

std::size_t bm3d_scratch_floats(int block, int group, bool wiener) {
    // Only the block kernel with its cubes in global memory keeps the
    // reference cube outside the block.
    const bool global = !warp_serves(block, group, wiener) && cube_bytes(block, group, wiener) > kCubeSharedBytes;
    return wiener && global ? static_cast<std::size_t>(group) * block * block : 0;
}

bool bm3d_fuses(int block, int group, bool wiener) {
    // Admitted on paired measurements against ordered aggregation. The block
    // kernel fuses whenever its cube is staged in shared memory. The warp
    // kernel fuses except where the extra code cost more than it saved.
    if (!warp_serves(block, group, wiener)) return cube_bytes(block, group, wiener) <= kCubeSharedBytes;
    const int shape = block * 100 + group;
    if (shape == 1632) return false;                 // 512 samples per lane: no registers left
    if (shape == 1616 && !wiener) return false;      // 0.93x
    if (shape == 816 && wiener) return false;        // 0.95x
    return true;
}

void bm3d_filter_groups(const Bm3dGroupArgs& args, cudaStream_t stream) {
    if (args.batch <= 0) return;
    if (!launch_warps(args, stream) && !launch_shapes(args, stream)) {
        throw std::logic_error("nss_cuda: unsupported BM3D block or group size");
    }
    NSS_CUDA_CHECK_LAUNCH();
}

}  // namespace nss_cuda
