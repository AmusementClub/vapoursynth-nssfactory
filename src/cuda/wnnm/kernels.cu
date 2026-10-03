// SPDX-License-Identifier: GPL-2.0-only
#include "cuda/wnnm/kernels.hpp"
#include "cuda/runtime/error.hpp"

#include <cstddef>

namespace nss_cuda {
namespace {

// One thread per group with fixed loop bounds, so every step has a fixed
// order (run-to-run identical) and the n x n work stays out of global memory.
// N is the column capacity. At N = 8 (the default group size) the Gram matrix
// and eigenvectors live in registers; larger capacities keep them in shared
// memory (a per-thread stack frame of that size would make the driver reserve
// unbudgeted device memory). The image rows are streamed twice (Gram, then
// reconstruction). Columns beyond the group's actual count are zero and
// contribute zero singular values.
template <int N>
constexpr int wnnm_block_threads() {
    return N <= 8 ? 64 : 8192 / (2 * N * N);  // 32 KiB of shared memory per block
}
template <int N>
__global__ void wnnm_group_kernel(WnnmGroupArgs a) {
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
#pragma unroll
    for (int i = 0; i < N * N; ++i) {
        g[i] = 0.f;
        v[i] = (i / N == i % N) ? 1.f : 0.f;
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
        if (a.residual) {
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

    // Cyclic Jacobi with branch-free rotations.
    for (int sweep = 0; sweep < 16; ++sweep) {
        float off = 0.f, diag = 0.f;
#pragma unroll
        for (int i = 0; i < N; ++i) {
            diag = fmaf(g[i * N + i], g[i * N + i], diag);
#pragma unroll
            for (int j = i + 1; j < N; ++j) off = fmaf(g[i * N + j], g[i * N + j], off);
        }
        if (off <= 1e-13f * diag || off <= 1e-30f) break;
#pragma unroll(N <= 8 ? N : 1)
        for (int p = 0; p < N - 1; ++p) {
#pragma unroll(N <= 8 ? N : 1)
            for (int q = p + 1; q < N; ++q) {
                const float apq = g[p * N + q];
                const bool rotate = apq != 0.f;
                const float theta = (g[q * N + q] - g[p * N + p]) / (2.f * (rotate ? apq : 1.f));
                const float t = copysignf(1.f, theta) / (fabsf(theta) + sqrtf(fmaf(theta, theta, 1.f)));
                const float c = rotate ? rsqrtf(fmaf(t, t, 1.f)) : 1.f;
                const float sn = rotate ? t * c : 0.f;
#pragma unroll
                for (int k = 0; k < N; ++k) {
                    const float gkp = g[k * N + p], gkq = g[k * N + q];
                    g[k * N + p] = c * gkp - sn * gkq;
                    g[k * N + q] = sn * gkp + c * gkq;
                }
#pragma unroll
                for (int k = 0; k < N; ++k) {
                    const float gpk = g[p * N + k], gqk = g[q * N + k];
                    g[p * N + k] = c * gpk - sn * gqk;
                    g[q * N + k] = sn * gpk + c * gqk;
                }
#pragma unroll
                for (int k = 0; k < N; ++k) {
                    const float vkp = v[k * N + p], vkq = v[k * N + q];
                    v[k * N + p] = c * vkp - sn * vkq;
                    v[k * N + q] = sn * vkp + c * vkq;
                }
            }
        }
    }

    float s[N];
    int order[N];
#pragma unroll
    for (int j = 0; j < N; ++j) {
        const float lambda = g[j * N + j];
        s[j] = lambda > 0.f ? sqrtf(lambda) : 0.f;
        order[j] = j;
    }
    for (int i = 1; i < N; ++i) {
        const int key = order[i];
        int j = i - 1;
        while (j >= 0 && s[order[j]] < s[key]) {
            order[j + 1] = order[j];
            --j;
        }
        order[j + 1] = key;
    }
    const int rank = min(area, n);
    const float constant = 8.f * sqrtf(2.f * static_cast<float>(n)) * a.sigma * a.sigma;
    const int start = min(a.residual ? 0 : 1, rank);
    float gain[N];
#pragma unroll
    for (int j = 0; j < N; ++j) gain[j] = 0.f;
    for (int k = 0; k < start; ++k) gain[order[k]] = 1.f;
    int kept = start;
    for (; kept < rank; ++kept) {
        const float sv = s[order[kept]];
        const float tmp = sv * sv - constant;
        if (!(tmp > 0.f)) break;
        gain[order[kept]] = ((sv + sqrtf(tmp)) * 0.5f) / sv;
    }
    // M = V diag(gain) V^T (into g), then X = A M row by row.
#pragma unroll
    for (int i = 0; i < N; ++i) {
#pragma unroll
        for (int j = 0; j < N; ++j) {
            float sum = 0.f;
#pragma unroll
            for (int k = 0; k < N; ++k) sum = fmaf(v[i * N + k] * gain[k], v[j * N + k], sum);
            g[i * N + j] = sum;
        }
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
        const float mu = a.residual ? sum / static_cast<float>(n) : 0.f;
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
    const float weight = a.adaptive && kept > 0 ? 1.f / static_cast<float>(kept) : 1.f;
    for (int j = 0; j < a.group; ++j) {
        a.patches[static_cast<long long>(r) * a.group + j] =
            j < n ? AggregatePatch{m[j].x, m[j].y, m[j].t, weight} : AggregatePatch{0, 0, -1, 0.f};
    }
}

}  // namespace

void wnnm_filter_groups(const WnnmGroupArgs& args, cudaStream_t stream) {
    if (args.batch <= 0) return;
    const auto launch = [&](auto kernel, int threads, int capacity) {
        const std::size_t shared = capacity <= 8 ? 0 : static_cast<std::size_t>(threads) * 2 * capacity * capacity * sizeof(float);
        kernel<<<(args.batch + threads - 1) / threads, threads, shared, stream>>>(args);
    };
    if (args.group <= 8) launch(wnnm_group_kernel<8>, wnnm_block_threads<8>(), 8);
    else if (args.group <= 16) launch(wnnm_group_kernel<16>, wnnm_block_threads<16>(), 16);
    else launch(wnnm_group_kernel<32>, wnnm_block_threads<32>(), 32);  // nss::kWnnmMaxGroup
    NSS_CUDA_CHECK_LAUNCH();
}

}  // namespace nss_cuda
