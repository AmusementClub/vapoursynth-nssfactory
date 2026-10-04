// SPDX-License-Identifier: GPL-2.0-only
#include "cuda/ncsr/kernels.hpp"
#include "cuda/common/gram_group.cuh"
#include "cuda/common/gram_shrink.cuh"

namespace nss_cuda {
namespace {

struct NcsrModel {
    float sigma;
    __device__ bool center() const { return true; }
    template <int N>
    __device__ float finish(float* g, const float* v, const DeviceMatch* matches, int n, int area, bool codes) const {
        float s[N];
        int order[N];
        singular_order<N>(g, s, order);
        // Column weights exp(-distance / h) (nss::ncsr_group_weights).
        constexpr float epsilon = 1e-12f;
        const float h = fmaxf(2.f * static_cast<float>(area) * sigma * sigma, epsilon);
        float weight[N];
        float sum = 0.f;
#pragma unroll
        for (int j = 0; j < N; ++j) {
            weight[j] = j < n ? expf(-(j > 0 ? matches[j].dist : 0.f) / h) : 0.f;
            sum += weight[j];
        }
        if (!(sum > 0.f)) {
#pragma unroll
            for (int j = 0; j < N; ++j) weight[j] = j < n ? 1.f : 0.f;
            sum = static_cast<float>(n);
        }
        const float inverse = 1.f / sum;
        const float sigma2 = sigma * sigma;
#pragma unroll
        for (int i = 0; i < N * N; ++i) g[i] = 0.f;
        const int rank = min(area, n);
        for (int k = 0; k < rank; ++k) {
            const int i = order[k];
            const float si = s[i];
            if (!(si > 0.f) || !(sigma > 0.f)) continue;
            // Code row s_i V[:, i]: weighted mean and deviation, then the
            // centralized soft threshold; stored relative to s_i.
            float code[N];
            float mean = 0.f;
#pragma unroll
            for (int j = 0; j < N; ++j) {
                code[j] = j < n ? si * v[j * N + i] : 0.f;
                mean = fmaf(weight[j], code[j], mean);
            }
            mean *= inverse;
            float variance = 0.f;
#pragma unroll
            for (int j = 0; j < N; ++j) {
                const float error = code[j] - mean;
                variance = fmaf(weight[j] * error, error, variance);
            }
            const float tau = 2.8284271247461903f * sigma2 / (sqrtf(variance * inverse) + epsilon);
            const float inv_s = 1.f / si;
#pragma unroll
            for (int j = 0; j < N; ++j) {
                const float d = code[j] - mean;
                const float a = fabsf(d);
                code[j] = j < n ? ((a > tau ? copysignf(a - tau, d) : 0.f) + mean) * inv_s : 0.f;
            }
            if (codes) {
                for (int j = 0; j < N; ++j) g[i * N + j] = code[j];
                continue;
            }
#pragma unroll
            for (int row = 0; row < N; ++row) {
                const float vi = v[row * N + i];
#pragma unroll
                for (int j = 0; j < N; ++j) g[row * N + j] = fmaf(vi, code[j], g[row * N + j]);
            }
        }
        if (!(sigma > 0.f)) {
            // No shrinkage: the group is returned unchanged (M = I, C = V^T).
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) g[i * N + j] = codes ? v[j * N + i] : (i == j ? 1.f : 0.f);
            }
        }
        return 1.f;
    }
};

}  // namespace

void ncsr_filter_groups(const NcsrGroupArgs& args, cudaStream_t stream) {
    const GramGroupArgs shell{args.src, args.pitch, args.matches, args.counts, args.batch,
                              args.block, args.group, args.values, args.patches};
    launch_gram_groups(shell, NcsrModel{args.sigma}, stream);
}

}  // namespace nss_cuda
