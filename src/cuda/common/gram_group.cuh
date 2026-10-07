// SPDX-License-Identifier: GPL-2.0-only
// Group kernel shell for filters whose result is the (optionally row-centered)
// m x n group matrix A times an n x n matrix M derived from the Gram matrix
// A^T A (WNNM: singular-value shrinkage; NCSR: centralized code shrinkage).
//
// Columns beyond the group's actual count are zero and contribute zero
// singular values. Every step has a fixed order, so the result is run-to-run
// identical. The image rows are streamed twice (Gram, then reconstruction).
//
// With GramGroupArgs::fused the kernel adds its weighted patches to the
// fixed-point accumulators itself (exact integer sums, any order) and writes
// neither values nor patch records.
//
// Up to 8 columns (the default group size): one thread per group, with the
// Gram matrix and eigenvectors in registers and a cyclic Jacobi sweep.
// Above 8: N threads per group (two groups to a block of 32 at N = 16), with
// the two N x N matrices in shared memory and the round-robin parallel Jacobi
// of block_jacobi.cuh.
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
#include "cuda/common/fixed_accumulate.cuh"
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
    FixedTarget fused{};        // with num set: aggregate here; values and patches are unused
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
    const bool fused = a.fused.num != nullptr;
    long long cell[N];
#pragma unroll
    for (int j = 0; j < N; ++j) cell[j] = fused && j < n ? fixed_patch_cell(a.fused, m[j].t, m[j].y, m[j].x) : -1;
    // X = A M row by row.
    for (int p = 0; p < area; ++p) {
        const int offset = (p / a.block) * a.pitch + p % a.block;
        const int target = (p / a.block) * a.fused.pitch + p % a.block;
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
            if (fused) {
                if (cell[c] >= 0) fixed_add(a.fused.num, cell[c] + target, weight * (acc + mu));
            } else if (c < n) {
                x[c * area + p] = acc + mu;
            }
        }
    }
    if (fused) {
        // The den once per patch, at its first cell: the finish box-sums it.
#pragma unroll
        for (int j = 0; j < N; ++j) {
            if (cell[j] >= 0) fixed_add(a.fused.den, cell[j], weight);
        }
        return;
    }
    for (int j = 0; j < a.group; ++j) {
        a.patches[static_cast<long long>(r) * a.group + j] =
            j < n ? AggregatePatch{m[j].x, m[j].y, m[j].t, weight} : AggregatePatch{0, 0, -1, 0.f};
    }
}

// N threads per group and 32 / N groups per block, so that the warp is full
// (a block's groups share the barriers and nothing else). A block's last
// groups may lie beyond the batch: they run as empty groups and store nothing.
template <int N, class Model>
__global__ void gram_block_kernel(GramGroupArgs a, Model model) {
    constexpr int kGroups = 32 / N;
    __shared__ float g_all[kGroups][N * N], v_all[kGroups][N * N], cs_all[kGroups][N + 2], weight_all[kGroups];
    __shared__ int flag;
    const int slot = threadIdx.x / N, lane = threadIdx.x % N;
    float* const g = g_all[slot];
    float* const v = v_all[slot];
    float* const cs = cs_all[slot];
    float& weight = weight_all[slot];
    const bool live = static_cast<int>(blockIdx.x) * kGroups + slot < a.batch;
    const int r = live ? blockIdx.x * kGroups + slot : a.batch - 1;
    const int area = a.block * a.block;
    const int n = live ? min(a.counts[r], a.group) : 0;
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
    block_jacobi<float>(g, v, N, cs, &flag, true, lane, N);
    if (lane == 0 && live) weight = model.template finish<N>(g, v, m, n, area, true);
    __syncthreads();
    const bool fused = a.fused.num != nullptr;
    const long long cell =
        fused && lane < n ? fixed_patch_cell(a.fused, m[lane].t, m[lane].y, m[lane].x) : -1;
    // X = A M: output column `lane`, with that column of M = V C.
    if (lane < n && (!fused || cell >= 0)) {
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
            if (fused) {
                fixed_add(a.fused.num, cell + (p / a.block) * a.fused.pitch + p % a.block, weight * (out + mu));
            } else {
                x[lane * area + p] = out + mu;
            }
        }
        if (fused) fixed_add(a.fused.den, cell, weight);
    }
    if (fused || !live) return;
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
        gram_block_kernel<16, Model><<<(batch + 1) / 2, 32, 0, stream>>>(args, model);
    } else {
        gram_block_kernel<32, Model><<<batch, 32, 0, stream>>>(args, model);  // nss::kWnnmMaxGroup
    }
    NSS_CUDA_CHECK_LAUNCH();
}

}  // namespace nss_cuda
