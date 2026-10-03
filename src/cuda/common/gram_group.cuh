// SPDX-License-Identifier: GPL-2.0-only
// Group kernel shell for filters whose result is the (optionally row-centered)
// m x n group matrix A times an n x n matrix M derived from the Gram matrix
// A^T A (WNNM: singular-value shrinkage; NCSR: centralized code shrinkage).
//
// One thread per group with fixed loop bounds, so every step has a fixed
// order (run-to-run identical) and the n x n work stays out of global memory.
// N is the column capacity. At N = 8 (the default group size) the Gram matrix
// and eigenvectors live in registers; larger capacities keep them in shared
// memory (a per-thread stack frame of that size would make the driver reserve
// unbudgeted device memory). The image rows are streamed twice (Gram, then
// reconstruction). Columns beyond the group's actual count are zero and
// contribute zero singular values.
//
// Model requirements (a trivially copyable struct passed to the kernel):
//   bool center() const;  // subtract each row's mean over the group
//   template <int N> float transform(float* g, float* v, const DeviceMatch* matches, int n, int area) const;
//     g holds the Gram matrix on entry and M on return (X = A M, row-vector
//     convention: x_c = sum_k a_k M[k * N + c]); v is N x N workspace; the
//     return value is the aggregation weight of the group's patches.
#pragma once

#include "cuda/common/aggregate.hpp"
#include "cuda/common/match.hpp"
#include "cuda/runtime/error.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace nss_cuda {

struct GramGroupArgs {
    const float* const* src;    // device array of per-frame plane pointers (DeviceMatch::t)
    int pitch;
    const DeviceMatch* matches; // batch * group (group-strided)
    const int* counts;
    int batch;
    int block;
    int group;                  // <= 32
    float* values;              // batch * group * block^2
    AggregatePatch* patches;    // batch * group
};

template <int N>
constexpr int gram_block_threads() {
    return N <= 8 ? 64 : 8192 / (2 * N * N);  // 32 KiB of shared memory per block
}

template <int N, class Model>
__global__ void gram_group_kernel(GramGroupArgs a, Model model) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= a.batch) return;
    const int area = a.block * a.block;
    const int n = min(a.counts[r], a.group);
    const DeviceMatch* m = a.matches + static_cast<long long>(r) * a.group;
    float* x = a.values + static_cast<long long>(r) * a.group * area;
    const float* col[N];
#pragma unroll
    for (int j = 0; j < N; ++j) {
        const DeviceMatch& mj = m[j < n ? j : 0];
        col[j] = a.src[mj.t] + static_cast<long long>(mj.y) * a.pitch + mj.x;
    }
    extern __shared__ float shared[];
    float g_local[N <= 8 ? N * N : 1], v_local[N <= 8 ? N * N : 1];
    float* const g = N <= 8 ? g_local : shared + threadIdx.x * 2 * N * N;
    float* const v = N <= 8 ? v_local : g + N * N;
    const bool center = model.center();
#pragma unroll
    for (int i = 0; i < N * N; ++i) g[i] = 0.f;
    for (int p = 0; p < area; ++p) {
        const int offset = (p / a.block) * a.pitch + p % a.block;
        float row[N];
        float sum = 0.f;
#pragma unroll
        for (int j = 0; j < N; ++j) {
            row[j] = j < n ? col[j][offset] : 0.f;
            sum += row[j];
        }
        if (center) {
            const float mu = sum / static_cast<float>(n);
#pragma unroll
            for (int j = 0; j < N; ++j) row[j] = j < n ? row[j] - mu : 0.f;
        }
#pragma unroll
        for (int j = 0; j < N; ++j) {
#pragma unroll
            for (int k = 0; k <= j; ++k) g[j * N + k] = fmaf(row[j], row[k], g[j * N + k]);
        }
    }
#pragma unroll
    for (int j = 0; j < N; ++j) {
#pragma unroll
        for (int k = 0; k < j; ++k) g[k * N + j] = g[j * N + k];
    }
    const float weight = model.template transform<N>(g, v, m, n, area);
    // X = A M row by row.
    for (int p = 0; p < area; ++p) {
        const int offset = (p / a.block) * a.pitch + p % a.block;
        float row[N];
        float sum = 0.f;
#pragma unroll
        for (int j = 0; j < N; ++j) {
            row[j] = j < n ? col[j][offset] : 0.f;
            sum += row[j];
        }
        const float mu = center ? sum / static_cast<float>(n) : 0.f;
#pragma unroll
        for (int j = 0; j < N; ++j) row[j] = j < n ? row[j] - mu : 0.f;
#pragma unroll
        for (int c = 0; c < N; ++c) {
            float acc = 0.f;
#pragma unroll
            for (int k = 0; k < N; ++k) acc = fmaf(row[k], g[k * N + c], acc);
            if (c < n) x[c * area + p] = acc + mu;
        }
    }
    for (int j = 0; j < a.group; ++j) {
        a.patches[static_cast<long long>(r) * a.group + j] =
            j < n ? AggregatePatch{m[j].x, m[j].y, m[j].t, weight} : AggregatePatch{0, 0, -1, 0.f};
    }
}

template <class Model>
void launch_gram_groups(const GramGroupArgs& args, const Model& model, cudaStream_t stream) {
    if (args.batch <= 0) return;
    const auto shared = [](int threads, int capacity) {
        return capacity <= 8 ? std::size_t{0} : static_cast<std::size_t>(threads) * 2 * capacity * capacity * sizeof(float);
    };
    const auto blocks = [&](int threads) { return static_cast<unsigned>((args.batch + threads - 1) / threads); };
    if (args.group <= 8) {
        const int t = gram_block_threads<8>();
        gram_group_kernel<8, Model><<<blocks(t), t, shared(t, 8), stream>>>(args, model);
    } else if (args.group <= 16) {
        const int t = gram_block_threads<16>();
        gram_group_kernel<16, Model><<<blocks(t), t, shared(t, 16), stream>>>(args, model);
    } else {
        const int t = gram_block_threads<32>();  // nss::kWnnmMaxGroup
        gram_group_kernel<32, Model><<<blocks(t), t, shared(t, 32), stream>>>(args, model);
    }
    NSS_CUDA_CHECK_LAUNCH();
}

}  // namespace nss_cuda
