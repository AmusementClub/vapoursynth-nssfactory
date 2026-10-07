// SPDX-License-Identifier: GPL-2.0-only
#include "cuda/nlm/kernels.hpp"
#include "cuda/runtime/error.hpp"

#include <cfloat>
#include <cstddef>

namespace nss_cuda {
namespace {

__device__ __forceinline__ int clampi(int v, int lo, int hi) { return min(max(v, lo), hi); }

template <class Index>
__device__ __forceinline__ float pair_distance(NlmDistance mode, const NlmPlanes& c, Index ci, const NlmPlanes& n,
                                               Index ni) {
    switch (mode) {
    case NlmDistance::Luma: {
        const float t = c.p[0][ci] - n.p[0][ni];
        return 3.0f * t * t;
    }
    case NlmDistance::Chroma: {
        const float e1 = c.p[0][ci] - n.p[0][ni];
        const float e2 = c.p[1][ci] - n.p[1][ni];
        return 1.5f * (e1 * e1 + e2 * e2);
    }
    case NlmDistance::Yuv: {
        const float e0 = c.p[0][ci] - n.p[0][ni];
        const float e1 = c.p[1][ci] - n.p[1][ni];
        const float e2 = c.p[2][ci] - n.p[2][ni];
        return e0 * e0 + e1 * e1 + e2 * e2;
    }
    default: {
        const float u1 = c.p[0][ci], v1 = n.p[0][ni];
        const float u2 = c.p[1][ci], v2 = n.p[1][ni];
        const float u3 = c.p[2][ci], v3 = n.p[2][ni];
        const float m_red = (u1 + v1) / 6.0f;
        return (2.0f / 3.0f + m_red) * (u1 - v1) * (u1 - v1) + (4.0f / 3.0f) * (u2 - v2) * (u2 - v2) +
               (1.0f - m_red) * (u3 - v3) * (u3 - v3);
    }
    }
}

// exp(x) for x <= 0 as the CPU takes it (Highway FastExp without subnormals):
// the same operations in the same order, so the weights agree with the CPU's
// to the rounding of their arguments.
__device__ __forceinline__ float fast_exp(float x) {
    x = fmaxf(x, -88.0f);
    const float offset = __int_as_float(__float_as_int(0.5f) | (__float_as_int(x) & static_cast<int>(0x80000000u)));
    const int q = static_cast<int>(fmaf(x, 1.442695040888963407359924681f, offset));
    const float r = fmaf(static_cast<float>(q), -0.69314718056f, x);
    const float r2 = r * r;
    const float term0 = fmaf(0.99996228117046825901f, r, 1.0000001510806224569f);
    const float term1 = fmaf(0.16792157982876812494f, r, 0.49998365704575670199f);
    const float term2 = fmaf(0.041959439862987071845f, r2, term1);
    return fmaf(term2, r2, term0) * __int_as_float((q + 0x7f) << 23);
}

__global__ void hsum_kernel(NlmPlanes center, NlmPlanes neighbor, NlmDistance mode, int ox, int oy, int s, int width,
                            int height, float* hsum) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const long long row = static_cast<long long>(y) * width;
    const long long nrow = static_cast<long long>(clampi(y + oy, 0, height - 1)) * width;
    float sum = 0.f;
    for (int j = -s; j <= s; ++j) {
        const int xx = clampi(x + j, 0, width - 1);
        sum += pair_distance(mode, center, row + xx, neighbor, nrow + clampi(xx + ox, 0, width - 1));
    }
    hsum[row + x] = sum;
}

__device__ __forceinline__ float welsch(const float* hsum, int x, int y, int s, float h2_inv_norm, int width,
                                        int height) {
    float sum = 0.f;
    for (int k = -s; k <= s; ++k) sum += hsum[static_cast<long long>(clampi(y + k, 0, height - 1)) * width + x];
    return fast_exp(-sum * h2_inv_norm);
}

__global__ void accumulate_kernel(NlmAccumArgs a) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= a.width || y >= a.height) return;
    const long long i = static_cast<long long>(y) * a.width + x;
    const int mx = clampi(x - a.ox, 0, a.width - 1);
    const int my = clampi(y - a.oy, 0, a.height - 1);
    const long long mq = static_cast<long long>(my) * a.width + mx;
    const long long pq = static_cast<long long>(clampi(y + a.oy, 0, a.height - 1)) * a.width +
                         clampi(x + a.ox, 0, a.width - 1);
    const float u4 = welsch(a.hsum_bwd, x, y, a.s, a.h2_inv_norm, a.width, a.height);
    const float u4_mq = welsch(a.hsum_fwd, mx, my, a.s, a.h2_inv_norm, a.width, a.height);
    a.weight[i] += u4 + u4_mq;
    a.max_weight[i] = fmaxf(fmaxf(u4, u4_mq), a.max_weight[i]);
    for (int c = 0; c < a.channels; ++c) {
        a.wdst[c][i] += u4 * a.src_bwd.p[c][pq] + u4_mq * a.src_fwd.p[c][mq];
    }
}

__global__ void fill_kernel(float* data, float value, long long count) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) data[i] = value;
}

__global__ void finish_kernel(const float* src, const float* weight, const float* max_weight, const float* wdst,
                              float wref, long long count, float* out) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float mul = wref * max_weight[i];
    out[i] = (mul * src[i] + wdst[i]) / (mul + weight[i]);
}

// Row of element e in a region `width` wide, with inverse = 1 / width. Exact
// for the element counts of a tile (below 2^16).
__device__ __forceinline__ int element_row(int e, float inverse) {
    return static_cast<int>((static_cast<float>(e) + 0.5f) * inverse);
}

// Sum of 2s + 1 values `stride` apart, in ascending order. S >= 0 fixes s at
// compile time (the loop unrolls); S < 0 takes it at run time.
template <int S>
__device__ __forceinline__ float box_sum(const float* first, int stride, int s) {
    const int taps = S >= 0 ? 2 * S + 1 : 2 * s + 1;
    float sum = 0.f;
    for (int t = 0; t < taps; ++t) sum += first[t * stride];
    return sum;
}

// Weights exp(-box sum * h2_inv_norm) of the distance map
// D(center(x, y), neighbor(clamp(x + ox), clamp(y + oy))) for the pixels
// [rx0, rx1] x [ry0, ry1] (inside the image) into wt, row by row. dm takes the
// distance map of the region with a halo of s, hs its row sums; both are
// indexed by the coordinate before clamping and hold the value at the clamped
// coordinate, so that the sums read them without clamping. The lanes of a
// warp read consecutive words: rows laid out for the sums over rows to read
// consecutive words instead were 1.7 times slower (bank conflicts). Every
// thread of the block calls it; it ends behind a barrier.
template <int NT, int S>
__device__ __forceinline__ void tile_weights(float* dm, float* hs, float* wt, const NlmPlanes& center,
                                             const NlmPlanes& neighbor, NlmDistance mode, int ox, int oy, int s,
                                             float h2_inv_norm, int width, int height, int rx0, int ry0, int rx1,
                                             int ry1, int tid) {
    const int rw = rx1 - rx0 + 1, rh = ry1 - ry0 + 1;
    const int dw = rw + 2 * s, dh = rh + 2 * s;
    const float inv_dw = 1.0f / static_cast<float>(dw);
    const float inv_rw = 1.0f / static_cast<float>(rw);
    for (int e = tid; e < dw * dh; e += NT) {
        const int ey = element_row(e, inv_dw);
        const int px = clampi(rx0 - s + e - ey * dw, 0, width - 1), py = clampi(ry0 - s + ey, 0, height - 1);
        dm[e] = pair_distance(mode, center, py * width + px, neighbor,
                              clampi(py + oy, 0, height - 1) * width + clampi(px + ox, 0, width - 1));
    }
    __syncthreads();
    for (int e = tid; e < rw * dh; e += NT) {
        const int ey = element_row(e, inv_rw);
        hs[e] = box_sum<S>(dm + ey * dw + (e - ey * rw), 1, s);
    }
    __syncthreads();
    for (int e = tid; e < rw * rh; e += NT) wt[e] = fast_exp(-box_sum<S>(hs + e, rw, s) * h2_inv_norm);
    __syncthreads();
}

// Shared floats of a TW x TH tile: the distance map, its row sums and the
// weights of the largest region (the tile and its mirror, |offset| = a).
constexpr long long tile_floats(int tw, int th, int a, int s) {
    const long long rw = tw + a, rh = th + a;
    return (rw + 2LL * s) * (rh + 2LL * s) + rw * (rh + 2LL * s) + rw * rh;
}

// A block of TW * TH / PIX threads owns a TW x TH tile; a thread keeps the
// sums of PIX of its pixels.
template <int TW, int TH, int PIX, int S>
__global__ void tile_kernel(NlmTileArgs a) {
    constexpr int NT = TW * TH / PIX;
    extern __shared__ float shared[];
    const int tid = threadIdx.x;
    const int width = a.width, height = a.height, s = a.s;
    const int x0 = blockIdx.x * TW, y0 = blockIdx.y * TH;
    const int x1 = min(x0 + TW, width) - 1, y1 = min(y0 + TH, height) - 1;
    float* const dm = shared;
    float* const hs = dm + (TW + a.a + 2 * s) * (TH + a.a + 2 * s);
    float* const wt = hs + (TW + a.a) * (TH + a.a + 2 * s);

    int x[PIX], y[PIX];
    bool inside[PIX];
    float weight[PIX], max_weight[PIX], sum[PIX][3];
    for (int k = 0; k < PIX; ++k) {
        const int local = tid + k * NT;
        x[k] = x0 + local % TW;
        y[k] = y0 + local / TW;
        inside[k] = x[k] <= x1 && y[k] <= y1;
        weight[k] = 0.f;
        max_weight[k] = FLT_EPSILON;
        for (int c = 0; c < 3; ++c) sum[k][c] = 0.f;
    }
    const int span = 2 * a.a + 1;
    const NlmPlanes& center = a.ref[a.d];
    for (int i = -a.d; i <= 0; ++i) {
        const NlmPlanes& bwd = a.ref[a.d + i];
        const NlmPlanes& fwd = a.ref[a.d - i];
        for (int oy = -a.a; oy <= a.a; ++oy) {
            for (int ox = -a.a; ox <= a.a; ++ox) {
                if (i * span * span + oy * span + ox >= 0) continue;
                // The mirrored pixels of the tile, q = clamp(p - o).
                const int qx0 = clampi(x0 - ox, 0, width - 1), qx1 = clampi(x1 - ox, 0, width - 1);
                const int qy0 = clampi(y0 - oy, 0, height - 1), qy1 = clampi(y1 - oy, 0, height - 1);
                float u4[PIX], u4_mq[PIX];
                if (i == 0) {
                    // One map serves both: the weights of the tile and of its mirror.
                    const int rx0 = min(x0, qx0), rx1 = max(x1, qx1), ry0 = min(y0, qy0), ry1 = max(y1, qy1);
                    tile_weights<NT, S>(dm, hs, wt, center, center, a.distance, ox, oy, s, a.h2_inv_norm, width,
                                        height, rx0, ry0, rx1, ry1, tid);
                    const int rw = rx1 - rx0 + 1;
                    for (int k = 0; k < PIX; ++k) {
                        if (!inside[k]) continue;
                        const int mx = clampi(x[k] - ox, 0, width - 1), my = clampi(y[k] - oy, 0, height - 1);
                        u4[k] = wt[(y[k] - ry0) * rw + x[k] - rx0];
                        u4_mq[k] = wt[(my - ry0) * rw + mx - rx0];
                    }
                } else {
                    tile_weights<NT, S>(dm, hs, wt, center, bwd, a.distance, ox, oy, s, a.h2_inv_norm, width, height,
                                        x0, y0, x1, y1, tid);
                    for (int k = 0; k < PIX; ++k) {
                        if (inside[k]) u4[k] = wt[(y[k] - y0) * (x1 - x0 + 1) + x[k] - x0];
                    }
                    tile_weights<NT, S>(dm, hs, wt, fwd, center, a.distance, ox, oy, s, a.h2_inv_norm, width, height,
                                        qx0, qy0, qx1, qy1, tid);
                    for (int k = 0; k < PIX; ++k) {
                        if (!inside[k]) continue;
                        const int mx = clampi(x[k] - ox, 0, width - 1), my = clampi(y[k] - oy, 0, height - 1);
                        u4_mq[k] = wt[(my - qy0) * (qx1 - qx0 + 1) + mx - qx0];
                    }
                }
                for (int k = 0; k < PIX; ++k) {
                    if (!inside[k]) continue;
                    const int pq = clampi(y[k] + oy, 0, height - 1) * width + clampi(x[k] + ox, 0, width - 1);
                    const int mq = clampi(y[k] - oy, 0, height - 1) * width + clampi(x[k] - ox, 0, width - 1);
                    weight[k] += u4[k] + u4_mq[k];
                    max_weight[k] = fmaxf(fmaxf(u4[k], u4_mq[k]), max_weight[k]);
                    for (int c = 0; c < a.channels; ++c) {
                        sum[k][c] += u4[k] * a.src[a.d + i].p[c][pq] + u4_mq[k] * a.src[a.d - i].p[c][mq];
                    }
                }
            }
        }
    }
    for (int k = 0; k < PIX; ++k) {
        if (!inside[k]) continue;
        const int at = y[k] * width + x[k];
        const float mul = a.wref * max_weight[k];
        for (int c = 0; c < a.channels; ++c) {
            a.out[c][at] = (mul * a.src[a.d].p[c][at] + sum[k][c]) / (mul + weight[k]);
        }
    }
}

// 512 threads with two pixels each: 408 fps at 1080p with the defaults, where
// 1024 threads with one pixel gave 380 and 64 x 32 tiles the same 408 with
// twice the shared memory (RTX 5080).
constexpr int kTileWidth = 32, kTileHeight = 32, kTilePixels = 2;
// The shared memory every supported device grants without asking.
constexpr long long kTileSharedBytes = 48 * 1024;
// Patch radii with a kernel of their own (the box sums unroll).
constexpr int kTileFixedS = 8;

template <int TW, int TH, int PIX, int S>
void launch_shape_s(const NlmTileArgs& args, cudaStream_t stream) {
    const dim3 grid((args.width + TW - 1) / TW, (args.height + TH - 1) / TH);
    const std::size_t shared = static_cast<std::size_t>(tile_floats(TW, TH, args.a, args.s)) * sizeof(float);
    tile_kernel<TW, TH, PIX, S><<<grid, TW * TH / PIX, shared, stream>>>(args);
}

template <int TW, int TH, int PIX>
void launch_shape(const NlmTileArgs& args, cudaStream_t stream) {
    switch (args.s <= kTileFixedS ? args.s : -1) {
    case 0: launch_shape_s<TW, TH, PIX, 0>(args, stream); break;
    case 1: launch_shape_s<TW, TH, PIX, 1>(args, stream); break;
    case 2: launch_shape_s<TW, TH, PIX, 2>(args, stream); break;
    case 3: launch_shape_s<TW, TH, PIX, 3>(args, stream); break;
    case 4: launch_shape_s<TW, TH, PIX, 4>(args, stream); break;
    case 5: launch_shape_s<TW, TH, PIX, 5>(args, stream); break;
    case 6: launch_shape_s<TW, TH, PIX, 6>(args, stream); break;
    case 7: launch_shape_s<TW, TH, PIX, 7>(args, stream); break;
    case 8: launch_shape_s<TW, TH, PIX, 8>(args, stream); break;
    default: launch_shape_s<TW, TH, PIX, -1>(args, stream); break;
    }
}

dim3 grid2d(int width, int height) { return dim3((width + 31) / 32, (height + 7) / 8); }
const dim3 kBlock2d(32, 8);

}  // namespace

void nlm_distance_hsum(const NlmPlanes& center, const NlmPlanes& neighbor, NlmDistance distance, int ox, int oy,
                       int s, int width, int height, float* hsum, cudaStream_t stream) {
    hsum_kernel<<<grid2d(width, height), kBlock2d, 0, stream>>>(center, neighbor, distance, ox, oy, s, width, height,
                                                                hsum);
    NSS_CUDA_CHECK_LAUNCH();
}

void nlm_accumulate(const NlmAccumArgs& args, cudaStream_t stream) {
    accumulate_kernel<<<grid2d(args.width, args.height), kBlock2d, 0, stream>>>(args);
    NSS_CUDA_CHECK_LAUNCH();
}

void nlm_reset(float* weight, float* max_weight, float* const* wdst, int channels, int width, int height,
               cudaStream_t stream) {
    const long long count = static_cast<long long>(width) * height;
    NSS_CUDA_CHECK(cudaMemsetAsync(weight, 0, count * sizeof(float), stream));
    for (int c = 0; c < channels; ++c) NSS_CUDA_CHECK(cudaMemsetAsync(wdst[c], 0, count * sizeof(float), stream));
    fill_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(max_weight, FLT_EPSILON, count);
    NSS_CUDA_CHECK_LAUNCH();
}

void nlm_finish(const float* src, const float* weight, const float* max_weight, const float* wdst, float wref,
                int width, int height, float* out, cudaStream_t stream) {
    const long long count = static_cast<long long>(width) * height;
    finish_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(src, weight, max_weight, wdst, wref,
                                                                                  count, out);
    NSS_CUDA_CHECK_LAUNCH();
}

bool nlm_tile_supported(int d, int a, int s) {
    return 2 * d + 1 <= kNlmTileFrames &&
           tile_floats(kTileWidth, kTileHeight, a, s) * static_cast<long long>(sizeof(float)) <= kTileSharedBytes;
}

void nlm_tile(const NlmTileArgs& args, cudaStream_t stream) {
    launch_shape<kTileWidth, kTileHeight, kTilePixels>(args, stream);
    NSS_CUDA_CHECK_LAUNCH();
}

}  // namespace nss_cuda
