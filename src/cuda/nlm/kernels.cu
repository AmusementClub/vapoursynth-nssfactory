// SPDX-License-Identifier: GPL-2.0-only
#include "cuda/nlm/kernels.hpp"
#include "cuda/runtime/error.hpp"

#include <cfloat>

namespace nss_cuda {
namespace {

__device__ __forceinline__ int clampi(int v, int lo, int hi) { return min(max(v, lo), hi); }

__device__ __forceinline__ float pair_distance(NlmDistance mode, const NlmPlanes& c, long long ci, const NlmPlanes& n,
                                               long long ni) {
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
    return expf(-sum * h2_inv_norm);
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

}  // namespace nss_cuda
