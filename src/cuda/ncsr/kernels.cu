// SPDX-License-Identifier: GPL-2.0-only
#include "cuda/ncsr/kernels.hpp"
#include "cuda/common/gram_group.cuh"
#include "cuda/common/gram_shrink.cuh"

namespace nss_cuda {
namespace {

// The columns of v in the order of singular_order (singular value
// descending, index ascending among equal values), with s alongside, by an
// odd-even transposition network: every index is a constant, so that a
// caller holding v in registers keeps it there (an index taken from `order`
// would put v, and with it the whole Jacobi, in local memory).
template <int N>
__device__ __forceinline__ void sorted_columns(const float* g, const float* v, float* s, float* u) {
    int index[N];
#pragma unroll
    for (int j = 0; j < N; ++j) {
        const float lambda = g[j * N + j];
        s[j] = lambda > 0.f ? sqrtf(lambda) : 0.f;
        index[j] = j;
    }
#pragma unroll
    for (int e = 0; e < N * N; ++e) u[e] = v[e];
#pragma unroll
    for (int round = 0; round < N; ++round) {
#pragma unroll
        for (int a = round & 1; a + 1 < N; a += 2) {
            const int b = a + 1;
            const bool keep = s[a] > s[b] || (s[a] == s[b] && index[a] < index[b]);
            const float sa = s[a];
            const int ia = index[a];
            s[a] = keep ? sa : s[b];
            s[b] = keep ? s[b] : sa;
            index[a] = keep ? ia : index[b];
            index[b] = keep ? index[b] : ia;
#pragma unroll
            for (int row = 0; row < N; ++row) {
                const float ua = u[row * N + a];
                u[row * N + a] = keep ? ua : u[row * N + b];
                u[row * N + b] = keep ? u[row * N + b] : ua;
            }
        }
    }
}

struct NcsrModel {
    float sigma;
    // Centred groups: the tridiagonal eigen step is accurate (gram_group.cuh).
    static constexpr bool kDivideAndConquer = true;
    __device__ bool center() const { return true; }

    // Column weights exp(-distance / h) (nss::ncsr_group_weights); returns
    // the reciprocal of their sum.
    template <int N>
    __device__ float weights(const DeviceMatch* matches, int n, int area, float* weight) const {
        constexpr float epsilon = 1e-12f;
        const float h = fmaxf(2.f * static_cast<float>(area) * sigma * sigma, epsilon);
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
        return 1.f / sum;
    }

    // Code row s_i V[:, i] (column `column` of v scaled by si): weighted mean
    // and deviation, then the centralized soft threshold; stored relative to
    // s_i.
    template <int N>
    __device__ void code_row(const float* v, int column, float si, const float* weight, float inverse, int n,
                             float* code) const {
        constexpr float epsilon = 1e-12f;
        const float sigma2 = sigma * sigma;
        float mean = 0.f;
#pragma unroll
        for (int j = 0; j < N; ++j) {
            code[j] = j < n ? si * v[j * N + column] : 0.f;
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
    }

    // The block kernels' finish: lane i forms row i of C (eigenvalue i), the
    // rows finish's codes path forms one after the other on one thread, with
    // the same arithmetic. Row i is kept when its singular value is among the
    // rank largest (its place in singular_order's order: larger values, and
    // equal ones of smaller index, come first).
    template <int N>
    __device__ float finish_lane(float* g, const float* v, const DeviceMatch* matches, int n, int area,
                                 int lane) const {
        float weight[N];
        const float inverse = weights<N>(matches, n, area, weight);
        float si;
        const int place = singular_place<N>(g, lane, &si);
        __syncthreads();  // every eigenvalue is read before C overwrites g
        float* const row = g + lane * N;
        if (!(sigma > 0.f)) {
            // No shrinkage: the group is returned unchanged (C = V^T).
            for (int j = 0; j < N; ++j) row[j] = v[j * N + lane];
        } else if (place < min(area, n) && si > 0.f) {
            float code[N];
            code_row<N>(v, lane, si, weight, inverse, n, code);
#pragma unroll
            for (int j = 0; j < N; ++j) row[j] = code[j];
        } else {
#pragma unroll
            for (int j = 0; j < N; ++j) row[j] = 0.f;
        }
        return 1.f;
    }

    template <int N>
    __device__ float finish(float* g, const float* v, const DeviceMatch* matches, int n, int area, bool codes) const {
        float weight[N];
        const float inverse = weights<N>(matches, n, area, weight);
        const int rank = min(area, n);
        if (!codes) {
            // M = sum over the rank largest singular values of V[:, i] code_i,
            // in that order, with every index a constant (v in registers).
            float s[N], u[N * N];
            sorted_columns<N>(g, v, s, u);
#pragma unroll
            for (int i = 0; i < N * N; ++i) g[i] = 0.f;
#pragma unroll
            for (int k = 0; k < N; ++k) {
                if (k >= rank || !(s[k] > 0.f) || !(sigma > 0.f)) continue;
                float code[N];
                code_row<N>(u, k, s[k], weight, inverse, n, code);
#pragma unroll
                for (int row = 0; row < N; ++row) {
                    const float vi = u[row * N + k];
#pragma unroll
                    for (int j = 0; j < N; ++j) g[row * N + j] = fmaf(vi, code[j], g[row * N + j]);
                }
            }
            if (!(sigma > 0.f)) {
                // No shrinkage: the group is returned unchanged (M = I).
#pragma unroll
                for (int i = 0; i < N * N; ++i) g[i] = i / N == i % N ? 1.f : 0.f;
            }
            return 1.f;
        }
        // Block kernels (v in shared memory): C, row i for eigenvalue i.
        float s[N];
        int order[N];
        singular_order<N>(g, s, order);
        for (int i = 0; i < N * N; ++i) g[i] = 0.f;
        for (int k = 0; k < rank; ++k) {
            const int i = order[k];
            if (!(s[i] > 0.f) || !(sigma > 0.f)) continue;
            float code[N];
            code_row<N>(v, i, s[i], weight, inverse, n, code);
            for (int j = 0; j < N; ++j) g[i * N + j] = code[j];
        }
        if (!(sigma > 0.f)) {
            // No shrinkage: the group is returned unchanged (C = V^T).
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) g[i * N + j] = v[j * N + i];
            }
        }
        return 1.f;
    }
};

}  // namespace

void ncsr_filter_groups(const NcsrGroupArgs& args, cudaStream_t stream) {
    const GramGroupArgs shell{args.src, args.pitch, args.matches, args.counts, args.batch,
                              args.block, args.group, args.values, args.patches, args.fused, args.scratch};
    launch_gram_groups(shell, NcsrModel{args.sigma}, stream);
}

}  // namespace nss_cuda
