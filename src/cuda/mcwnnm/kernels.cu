// SPDX-License-Identifier: GPL-2.0-only
#include "cuda/mcwnnm/kernels.hpp"
#include "cuda/common/gram_shrink.cuh"
#include "cuda/runtime/error.hpp"

#include <cstddef>

namespace nss_cuda {
namespace {

// Threads per block: capacities above 8 keep three N x N matrices per thread
// in shared memory (24 KiB per block).
template <int N>
constexpr int mcwnnm_block_threads() {
    return N <= 8 ? 64 : 6144 / (3 * N * N);
}

// One thread per group with fixed loop bounds and order (run-to-run
// identical). N is the column capacity; columns beyond the group's count are
// zero and contribute zero singular values.
template <int N>
__global__ void mcwnnm_group_kernel(McwnnmGroupArgs a) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= a.batch) return;
    const int area = a.block * a.block;
    const int m = 3 * area;
    const int n = min(a.counts[r], a.group);
    const DeviceMatch* match = a.matches + static_cast<long long>(r) * a.group;
    const float* col[N];
#pragma unroll
    for (int j = 0; j < N; ++j) {
        const DeviceMatch& mj = match[j < n ? j : 0];
        col[j] = a.src[mj.t] + static_cast<long long>(mj.y) * a.pitch + mj.x;
    }
    // X and A are interleaved across the block's groups (element e of thread
    // l at e * lanes + l), so a warp's accesses to one element coalesce. The
    // input rows are re-read from the frames instead: that is cheaper than a
    // third interleaved array (measured).
    const int lanes = min(static_cast<int>(blockDim.x), a.batch - static_cast<int>(blockIdx.x * blockDim.x));
    float* x = a.scratch + static_cast<long long>(blockIdx.x) * blockDim.x * 2 * m * a.group + threadIdx.x;
    float* dual = x + static_cast<long long>(m) * a.group * lanes;
    const long long channel_values = static_cast<long long>(a.batch) * a.group * area;
    float* out = a.values + static_cast<long long>(r) * a.group * area;  // channel c at + c * channel_values

    extern __shared__ float shared[];
    float g_local[N <= 8 ? N * N : 1], v_local[N <= 8 ? N * N : 1], next_local[N <= 8 ? N * N : 1];
    float* const g = N <= 8 ? g_local : shared + threadIdx.x * 3 * N * N;
    float* const v = N <= 8 ? v_local : g + N * N;
    float* const next = N <= 8 ? next_local : v + N * N;

    // Channel weights (nss::channel_weight_diag): (smin / sigma_c)^2 with the
    // smallest positive sigma; a zero-sigma channel is pinned to its input.
    float smin = 0.f;
    for (int c = 0; c < 3; ++c) {
        if (a.sigma[c] > 0.f) smin = smin == 0.f ? a.sigma[c] : fminf(smin, a.sigma[c]);
    }
    float weight[3];
    for (int c = 0; c < 3; ++c) {
        const float sc = a.sigma[c] > 0.f ? a.sigma[c] : smin * 1e-6f;
        weight[c] = (smin / sc) * (smin / sc);
    }
    const float constant = 8.f * sqrtf(2.f * static_cast<float>(n)) * smin * smin;
    const int rank = min(m, n);
    const int start_k = a.residual ? 0 : 1;
    const float inv_n = 1.f / static_cast<float>(n);

    // Row i of the (centered) input and its mean.
    const auto load_row = [&](int i, float* row) {
        const int c = i / area, p = i % area;
        const long long offset = c * a.channel_step + static_cast<long long>(p / a.block) * a.pitch + p % a.block;
        float sum = 0.f;
#pragma unroll
        for (int j = 0; j < N; ++j) {
            row[j] = j < n ? col[j][offset] : 0.f;
            sum += row[j];
        }
        const float mean = a.residual ? sum * inv_n : 0.f;
#pragma unroll
        for (int j = 0; j < N; ++j) row[j] = j < n ? row[j] - mean : 0.f;
        return mean;
    };
    const auto add_gram = [&](float* gram, const float* t) {
#pragma unroll
        for (int j = 0; j < N; ++j) {
#pragma unroll
            for (int k = 0; k <= j; ++k) gram[j * N + k] = fmaf(t[j], t[k], gram[j * N + k]);
        }
    };

    // X0 = argmin with Z = A = 0; Temp0 = X0.
    float rho = a.rho;
#pragma unroll
    for (int i = 0; i < N * N; ++i) next[i] = 0.f;
    for (int i = 0; i < m; ++i) {
        float y[N];
        load_row(i, y);
        const float w = weight[i / area];
        const float scale = w / (w + 0.5f * rho);
#pragma unroll
        for (int j = 0; j < N; ++j) y[j] *= scale;
        for (int j = 0; j < n; ++j) {
            x[(i * a.group + j) * lanes] = y[j];
            dual[(i * a.group + j) * lanes] = 0.f;
        }
        add_gram(next, y);
    }

    int kept = 0;
    for (int it = 0; it < a.admm_iter; ++it) {
#pragma unroll
        for (int j = 0; j < N; ++j) {
#pragma unroll
            for (int k = 0; k <= j; ++k) {
                g[j * N + k] = next[j * N + k];
                g[k * N + j] = next[j * N + k];
            }
        }
#pragma unroll
        for (int i = 0; i < N * N; ++i) next[i] = 0.f;
        kept = gram_shrink<N>(g, v, rank, constant * (2.f / rho), start_k);
        const bool last = it + 1 == a.admm_iter;
        const float next_rho = fminf(1e4f, a.mu * rho);
        const float inv_rho = 1.f / rho, inv_next = 1.f / next_rho, half_next = 0.5f * next_rho;
        for (int i = 0; i < m; ++i) {
            float xr[N], ar[N], z[N];
#pragma unroll
            for (int j = 0; j < N; ++j) {
                xr[j] = j < n ? x[(i * a.group + j) * lanes] : 0.f;
                ar[j] = j < n ? dual[(i * a.group + j) * lanes] : 0.f;
            }
            // Z row = (X + A / rho) M.
#pragma unroll
            for (int c = 0; c < N; ++c) z[c] = 0.f;
#pragma unroll
            for (int k = 0; k < N; ++k) {
                const float t = fmaf(ar[k], inv_rho, xr[k]);
#pragma unroll
                for (int c = 0; c < N; ++c) z[c] = fmaf(t, g[k * N + c], z[c]);
            }
            float y[N];
            const float mean = load_row(i, y);
            if (last) {
                const int c = i / area, p = i % area;
                for (int j = 0; j < n; ++j) out[c * channel_values + j * area + p] = z[j] + mean;
                continue;
            }
            // Dual update, then the next X and Temp = X + A / next_rho.
            const float w = weight[i / area];
            const float reciprocal = 1.f / (w + half_next);
            float temp[N];
#pragma unroll
            for (int j = 0; j < N; ++j) {
                const float dual_new = fmaf(rho, xr[j] - z[j], ar[j]);
                const float x_new = (w * y[j] + half_next * (z[j] - dual_new * inv_next)) * reciprocal;
                ar[j] = dual_new;
                xr[j] = x_new;
                temp[j] = j < n ? fmaf(dual_new, inv_next, x_new) : 0.f;
            }
            for (int j = 0; j < n; ++j) {
                x[(i * a.group + j) * lanes] = xr[j];
                dual[(i * a.group + j) * lanes] = ar[j];
            }
            add_gram(next, temp);
        }
        rho = next_rho;
    }
    const float patch_weight = a.adaptive && kept > 0 ? 1.f / static_cast<float>(kept) : 1.f;
    for (int j = 0; j < a.group; ++j) {
        a.patches[static_cast<long long>(r) * a.group + j] =
            j < n ? AggregatePatch{match[j].x, match[j].y, match[j].t, patch_weight} : AggregatePatch{0, 0, -1, 0.f};
    }
}

}  // namespace

void mcwnnm_filter_groups(const McwnnmGroupArgs& args, cudaStream_t stream) {
    if (args.batch <= 0) return;
    const auto launch = [&](auto kernel, int threads, int capacity) {
        const std::size_t shared =
            capacity <= 8 ? 0 : static_cast<std::size_t>(threads) * 3 * capacity * capacity * sizeof(float);
        kernel<<<(args.batch + threads - 1) / threads, threads, shared, stream>>>(args);
    };
    if (args.group <= 8) launch(mcwnnm_group_kernel<8>, mcwnnm_block_threads<8>(), 8);
    else if (args.group <= 16) launch(mcwnnm_group_kernel<16>, mcwnnm_block_threads<16>(), 16);
    else launch(mcwnnm_group_kernel<32>, mcwnnm_block_threads<32>(), 32);  // nss::kWnnmMaxGroup
    NSS_CUDA_CHECK_LAUNCH();
}

}  // namespace nss_cuda
