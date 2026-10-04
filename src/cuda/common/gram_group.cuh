// SPDX-License-Identifier: GPL-2.0-only
// Group kernel shell for filters whose result is the (optionally row-centered)
// m x n group matrix A times an n x n matrix M derived from the Gram matrix
// A^T A (WNNM: singular-value shrinkage; NCSR: centralized code shrinkage).
//
// Columns beyond the group's actual count are zero and contribute zero
// singular values. Every step has a fixed order, so the result is run-to-run
// identical. The image rows are streamed twice (Gram, then reconstruction).
//
// Up to 8 columns (the default group size): one thread per group, with the
// Gram matrix and eigenvectors in registers and a cyclic Jacobi sweep.
// Above 8: one block of N threads per group, with the two N x N matrices in
// shared memory and the round-robin parallel Jacobi of block_jacobi.cuh.
// Thread j owns Gram row j and output column j. (One thread per group does
// not scale there: registers cannot hold the matrices, a per-thread stack
// frame of that size makes the driver reserve unbudgeted device memory, and
// per-thread shared memory caps a block at a handful of threads.)
//
// Model requirements (a trivially copyable struct passed to the kernel):
//   bool center() const;  // subtract each row's mean over the group
//   template <int N> float finish(float* g, const float* v, const DeviceMatch* matches, int n, int area,
//                                 bool codes) const;
//     on entry g carries the Gram matrix's eigenvalues on its diagonal and v
//     the eigenvectors (column i for eigenvalue i); on return g holds M
//     (X = A M, row-vector convention: x_c = sum_k a_k M[k * N + c]), or with
//     `codes` the factor C of M = V C, which the block kernel multiplies out
//     across its threads; the return value is the aggregation weight of the
//     group's patches.
#pragma once

#include "cuda/common/aggregate.hpp"
#include "cuda/common/block_jacobi.cuh"
#include "cuda/common/gram_shrink.cuh"
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
    float g[N * N], v[N * N];
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
    jacobi_eigen<N>(g, v);
    const float weight = model.template finish<N>(g, v, m, n, area, false);
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

// One block of N threads per group (blockIdx.x is the group).
template <int N, class Model>
__global__ void gram_block_kernel(GramGroupArgs a, Model model) {
    __shared__ float g[N * N], v[N * N], cs[N + 2], weight;
    __shared__ int flag;
    const int r = blockIdx.x, lane = threadIdx.x;
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
    const bool center = model.center();
    // Gram row `lane`.
    float acc[N];
#pragma unroll
    for (int k = 0; k < N; ++k) acc[k] = 0.f;
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
        const float mine = row[lane];
#pragma unroll
        for (int k = 0; k < N; ++k) acc[k] = fmaf(mine, row[k], acc[k]);
    }
#pragma unroll
    for (int k = 0; k < N; ++k) g[lane * N + k] = acc[k];
    __syncthreads();
    block_jacobi<float>(g, v, N, cs, &flag);
    if (lane == 0) weight = model.template finish<N>(g, v, m, n, area, true);
    __syncthreads();
    // X = A M: output column `lane`, with that column of M = V C.
    if (lane < n) {
#pragma unroll
        for (int k = 0; k < N; ++k) {
            float sum = 0.f;
#pragma unroll
            for (int i = 0; i < N; ++i) sum = fmaf(v[k * N + i], g[i * N + lane], sum);
            acc[k] = sum;
        }
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
            float out = 0.f;
#pragma unroll
            for (int k = 0; k < N; ++k) out = fmaf(k < n ? row[k] - mu : 0.f, acc[k], out);
            x[lane * area + p] = out + mu;
        }
    }
    for (int j = lane; j < a.group; j += N) {
        a.patches[static_cast<long long>(r) * a.group + j] =
            j < n ? AggregatePatch{m[j].x, m[j].y, m[j].t, weight} : AggregatePatch{0, 0, -1, 0.f};
    }
}

template <class Model>
void launch_gram_groups(const GramGroupArgs& args, const Model& model, cudaStream_t stream) {
    if (args.batch <= 0) return;
    const unsigned batch = static_cast<unsigned>(args.batch);
    if (args.group <= 8) {
        gram_group_kernel<8, Model><<<(batch + 63) / 64, 64, 0, stream>>>(args, model);
    } else if (args.group <= 16) {
        gram_block_kernel<16, Model><<<batch, 16, 0, stream>>>(args, model);
    } else {
        gram_block_kernel<32, Model><<<batch, 32, 0, stream>>>(args, model);  // nss::kWnnmMaxGroup
    }
    NSS_CUDA_CHECK_LAUNCH();
}

}  // namespace nss_cuda
