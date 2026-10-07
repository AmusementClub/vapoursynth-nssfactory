// SPDX-License-Identifier: GPL-2.0-only
#include "cuda/mcwnnm/kernels.hpp"
#include "cuda/common/block_jacobi.cuh"
#include "cuda/common/fixed_accumulate.cuh"
#include "cuda/common/gram_shrink.cuh"
#include "cuda/runtime/error.hpp"

namespace nss_cuda {
namespace {

// Up to 8 columns: one thread per group with fixed loop bounds and order
// (run-to-run identical). N is the column capacity; columns beyond the
// group's count are zero and contribute zero singular values.
//
// The ADMM runs on n x n matrices instead of the m x n iterates. Every step
// after the eigen step treats a row of X and A alone and linearly, with
// coefficients that depend only on the row's channel, so row i of channel c
// stays x_i = y_i P_c, a_i = y_i Q_c for the centered input row y_i. Then
//   Z_c = (P_c + Q_c / rho) M,
//   Q_c' = Q_c + rho (P_c - Z_c),
//   P_c' = (w_c I + rho' / 2 (Z_c - Q_c' / rho')) / (w_c + rho' / 2),
// and the Gram matrix of the next shrinkage input T_c' = P_c' + Q_c' / rho'
// is sum_c T_c'^T (Y_c^T Y_c) T_c'. The image rows are streamed twice: for
// the three channel Gram matrices, and for the output Y_c Z_c of the last
// iteration.
//
// An iteration's eigen step starts from the eigenvectors V of the one
// before: the Gram matrix is taken to V^T G V, which is nearly diagonal once
// the iterates settle, and the sweeps go on from V.
template <int N>
__global__ void mcwnnm_group_kernel(McwnnmGroupArgs a) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= a.batch) return;
    const int area = a.block * a.block;
    const int m = 3 * area;
    const int n = min(a.counts[r], a.group);
    const DeviceMatch* match = a.matches + static_cast<long long>(r) * a.group;
    const float* col[N];
    // Called again before the output pass, so the pointers do not hold
    // registers through the iterations.
    const auto locate = [&] {
#pragma unroll
        for (int j = 0; j < N; ++j) {
            const DeviceMatch& mj = match[j < n ? j : 0];
            col[j] = a.src[mj.t] + static_cast<long long>(mj.y) * a.pitch + mj.x;
        }
    };
    locate();
    // Per channel the input Gram matrix, P and Q, and the eigenvectors of
    // the last eigen step, each group x group and interleaved across the
    // block's groups (element e of thread l at e * lanes + l), so a warp's
    // accesses to one element coalesce.
    const int lanes = min(static_cast<int>(blockDim.x), a.batch - static_cast<int>(blockIdx.x * blockDim.x));
    const int square = a.group * a.group;
    float* const state = a.scratch + static_cast<long long>(blockIdx.x) * blockDim.x * 10 * square + threadIdx.x;
    const auto at = [&](int matrix, int c, int j, int k) -> float& {
        return state[static_cast<long long>((matrix * 3 + c) * square + j * a.group + k) * lanes];
    };
    enum { kGram = 0, kP = 1, kQ = 2, kVectors = 3 };  // kVectors: channel 0 only
    const long long channel_values = static_cast<long long>(a.batch) * a.group * area;
    float* out = a.values + static_cast<long long>(r) * a.group * area;  // channel c at + c * channel_values
    const bool fused = a.fused.num != nullptr;

    float g[N * N], v[N * N], next[N * N];  // registers

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

    // The channel Gram matrices; P_c = scale_c I and Q_c = 0 (X0 = argmin
    // with Z = A = 0), whose shrinkage input is X0 itself.
    float rho = a.rho;
#pragma unroll
    for (int i = 0; i < N * N; ++i) next[i] = 0.f;
    for (int c = 0; c < 3; ++c) {
#pragma unroll
        for (int i = 0; i < N * N; ++i) g[i] = 0.f;
        for (int p = 0; p < area; ++p) {
            float y[N];
            load_row(c * area + p, y);
#pragma unroll
            for (int j = 0; j < N; ++j) {
#pragma unroll
                for (int k = 0; k <= j; ++k) g[j * N + k] = fmaf(y[j], y[k], g[j * N + k]);
            }
        }
        const float scale = weight[c] / (weight[c] + 0.5f * rho);
#pragma unroll
        for (int j = 0; j < N; ++j) {
#pragma unroll
            for (int k = 0; k <= j; ++k) {
                const float value = g[j * N + k];
                next[j * N + k] = fmaf(scale * scale, value, next[j * N + k]);
                if (j < n) {
                    at(kGram, c, j, k) = value;
                    at(kGram, c, k, j) = value;
                    at(kP, c, j, k) = j == k ? scale : 0.f;
                    at(kP, c, k, j) = j == k ? scale : 0.f;
                    at(kQ, c, j, k) = 0.f;
                    at(kQ, c, k, j) = 0.f;
                }
            }
        }
    }

    int kept = 0;
    for (int it = 0; it < a.admm_iter; ++it) {
        if (it > 0) {
            // g = V^T S V for the symmetric S in next's lower triangle.
#pragma unroll
            for (int j = 0; j < N; ++j) {
#pragma unroll
                for (int k = 0; k < N; ++k) v[j * N + k] = j < n && k < n ? at(kVectors, 0, j, k) : j == k ? 1.f : 0.f;
            }
            // S V into g, then the lower triangle of V^T (S V) back into next.
#pragma unroll
            for (int i = 0; i < N; ++i) {
#pragma unroll
                for (int k = 0; k < N; ++k) {
                    float sum = 0.f;
#pragma unroll
                    for (int j = 0; j < N; ++j) sum = fmaf(next[i >= j ? i * N + j : j * N + i], v[j * N + k], sum);
                    g[i * N + k] = sum;
                }
            }
#pragma unroll
            for (int j = 0; j < N; ++j) {
#pragma unroll
                for (int k = 0; k <= j; ++k) {
                    float sum = 0.f;
#pragma unroll
                    for (int i = 0; i < N; ++i) sum = fmaf(v[i * N + j], g[i * N + k], sum);
                    next[j * N + k] = sum;
                }
            }
#pragma unroll
            for (int j = 0; j < N; ++j) {
#pragma unroll
                for (int k = 0; k <= j; ++k) {
                    g[j * N + k] = next[j * N + k];
                    g[k * N + j] = next[j * N + k];
                }
            }
            jacobi_sweeps<N>(g, v);
        } else {
#pragma unroll
            for (int j = 0; j < N; ++j) {
#pragma unroll
                for (int k = 0; k <= j; ++k) {
                    g[j * N + k] = next[j * N + k];
                    g[k * N + j] = next[j * N + k];
                }
            }
            jacobi_eigen<N>(g, v);
        }
#pragma unroll
        for (int i = 0; i < N * N; ++i) next[i] = 0.f;
        kept = gram_shrink_spectrum<N>(g, v, rank, constant * (2.f / rho), start_k);  // g = M
        const bool last = it + 1 == a.admm_iter;
        if (!last) {
#pragma unroll
            for (int j = 0; j < N; ++j) {
#pragma unroll
                for (int k = 0; k < N; ++k) {
                    if (j < n && k < n) at(kVectors, 0, j, k) = v[j * N + k];
                }
            }
        }
        const float next_rho = fminf(1e4f, a.mu * rho);
        const float inv_rho = 1.f / rho, inv_next = 1.f / next_rho, half_next = 0.5f * next_rho;
        for (int c = 0; c < 3; ++c) {
            const float w = weight[c];
            const float reciprocal = 1.f / (w + half_next);
            // Row by row: Z_c, then Q_c', P_c' and T_c' (in v); on the last
            // iteration v keeps Z_c.
#pragma unroll
            for (int row = 0; row < N; ++row) {
                float pr[N], qr[N], z[N];
#pragma unroll
                for (int j = 0; j < N; ++j) {
                    const bool inside = row < n && j < n;
                    pr[j] = inside ? at(kP, c, row, j) : 0.f;
                    qr[j] = inside ? at(kQ, c, row, j) : 0.f;
                    z[j] = 0.f;
                }
#pragma unroll
                for (int k = 0; k < N; ++k) {
                    const float t = fmaf(qr[k], inv_rho, pr[k]);
#pragma unroll
                    for (int j = 0; j < N; ++j) z[j] = fmaf(t, g[k * N + j], z[j]);
                }
#pragma unroll
                for (int j = 0; j < N; ++j) {
                    if (last) {
                        v[row * N + j] = z[j];
                        continue;
                    }
                    const float q_new = fmaf(rho, pr[j] - z[j], qr[j]);
                    const float p_new =
                        ((row == j ? w : 0.f) + half_next * (z[j] - q_new * inv_next)) * reciprocal;
                    const bool inside = row < n && j < n;
                    if (inside) {
                        at(kP, c, row, j) = p_new;
                        at(kQ, c, row, j) = q_new;
                    }
                    v[row * N + j] = inside ? fmaf(q_new, inv_next, p_new) : 0.f;
                }
            }
            if (last) {
                // Output rows Y_c Z_c.
                if (c == 0) locate();
                const float patch_weight = a.adaptive && kept > 0 ? 1.f / static_cast<float>(kept) : 1.f;
                unsigned long long* const num = a.fused.num + c * a.fused.channel_step;
                for (int p = 0; p < area; ++p) {
                    float y[N];
                    const float mean = load_row(c * area + p, y);
                    const int target = (p / a.block) * a.fused.pitch + p % a.block;
#pragma unroll
                    for (int j = 0; j < N; ++j) {
                        float sum = 0.f;
#pragma unroll
                        for (int k = 0; k < N; ++k) sum = fmaf(y[k], v[k * N + j], sum);
                        if (j >= n) continue;
                        if (fused) {
                            const long long cell = fixed_patch_cell(a.fused, match[j].t, match[j].y, match[j].x);
                            if (cell >= 0) fixed_add(num, cell + target, patch_weight * (sum + mean));
                        } else {
                            out[c * channel_values + j * area + p] = sum + mean;
                        }
                    }
                }
                if (fused) {
                    // The den once per patch, at its first cell: the finish box-sums it.
                    for (int j = 0; j < n; ++j) {
                        const long long cell = fixed_patch_cell(a.fused, match[j].t, match[j].y, match[j].x);
                        if (cell >= 0) fixed_add(a.fused.den + c * a.fused.channel_step, cell, patch_weight);
                    }
                }
                continue;
            }
            // next += T^T (G_c T), lower triangle.
#pragma unroll
            for (int row = 0; row < N; ++row) {
                float u[N];
#pragma unroll
                for (int j = 0; j < N; ++j) u[j] = 0.f;
#pragma unroll
                for (int k = 0; k < N; ++k) {
                    const float gram = row < n && k < n ? at(kGram, c, row, k) : 0.f;
#pragma unroll
                    for (int j = 0; j < N; ++j) u[j] = fmaf(gram, v[k * N + j], u[j]);
                }
#pragma unroll
                for (int j = 0; j < N; ++j) {
#pragma unroll
                    for (int k = 0; k <= j; ++k) next[j * N + k] = fmaf(v[row * N + j], u[k], next[j * N + k]);
                }
            }
        }
        rho = next_rho;
    }
    if (fused) return;
    const float patch_weight = a.adaptive && kept > 0 ? 1.f / static_cast<float>(kept) : 1.f;
    for (int j = 0; j < a.group; ++j) {
        a.patches[static_cast<long long>(r) * a.group + j] =
            j < n ? AggregatePatch{match[j].x, match[j].y, match[j].t, patch_weight} : AggregatePatch{0, 0, -1, 0.f};
    }
}

// Above 8 columns: N threads per group and 32 / N groups per block (a block's
// groups share the barriers and nothing else; those beyond the batch run as
// empty groups and store nothing), the same iteration with
// thread j owning row j of every matrix (its rows of the channel Gram
// matrices, P and Q in the scratch, stored by columns so that the threads'
// accesses to one column coalesce) and the round-robin parallel Jacobi of
// block_jacobi.cuh on the Gram matrix in shared memory.
template <int N>
__global__ void mcwnnm_block_kernel(McwnnmGroupArgs a) {
    constexpr int kGroups = 32 / N;
    __shared__ float g_all[kGroups][N * N], v_all[kGroups][N * N], t_all[kGroups][N * N], u_all[kGroups][N * N];
    __shared__ float cs_all[kGroups][N + 2];
    __shared__ int flag, kept_all[kGroups];
    const int slot = threadIdx.x / N, lane = threadIdx.x % N;
    float* const g = g_all[slot];
    float* const v = v_all[slot];
    float* const t = t_all[slot];
    float* const u = u_all[slot];
    float* const cs = cs_all[slot];
    int& kept = kept_all[slot];
    const bool live = static_cast<int>(blockIdx.x) * kGroups + slot < a.batch;
    const int r = live ? blockIdx.x * kGroups + slot : a.batch - 1;
    const int area = a.block * a.block;
    const int n = live ? min(a.counts[r], a.group) : 0;
    const DeviceMatch* match = a.matches + static_cast<long long>(r) * a.group;
    const float* col[N];
#pragma unroll
    for (int j = 0; j < N; ++j) {
        const DeviceMatch& mj = match[j < n ? j : 0];
        col[j] = a.src[mj.t] + static_cast<long long>(mj.y) * a.pitch + mj.x;
    }
    const int square = a.group * a.group;
    float* const state = a.scratch + static_cast<long long>(r) * 10 * square;
    enum { kGram = 0, kP = 1, kQ = 2 };
    // Element k of this thread's row of a channel's matrix (lane < n, k < n).
    const auto at = [&](int matrix, int c, int k) -> float& {
        return state[(matrix * 3 + c) * square + k * a.group + lane];
    };
    const long long channel_values = static_cast<long long>(a.batch) * a.group * area;
    float* out = a.values + static_cast<long long>(r) * a.group * area;
    const bool mine = lane < n;
    const bool fused = a.fused.num != nullptr;
    const long long cell =
        fused && mine ? fixed_patch_cell(a.fused, match[lane].t, match[lane].y, match[lane].x) : -1;

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
    const int rank = min(3 * area, n);
    const int start_k = a.residual ? 0 : 1;
    const float inv_n = 1.f / static_cast<float>(n);
    const auto load_row = [&](int c, int p, float* row) {
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

    // Row `lane` of the channel Gram matrices and of the first shrinkage
    // input's Gram matrix; P_c = scale_c I, Q_c = 0.
    float rho = a.rho;
    float next[N], acc[N];
#pragma unroll
    for (int k = 0; k < N; ++k) next[k] = 0.f;
    for (int c = 0; c < 3; ++c) {
#pragma unroll
        for (int k = 0; k < N; ++k) acc[k] = 0.f;
        for (int p = 0; p < area; ++p) {
            float y[N];
            load_row(c, p, y);
            const float own = y[lane];
#pragma unroll
            for (int k = 0; k < N; ++k) acc[k] = fmaf(own, y[k], acc[k]);
        }
        const float scale = weight[c] / (weight[c] + 0.5f * rho);
#pragma unroll
        for (int k = 0; k < N; ++k) {
            next[k] = fmaf(scale * scale, acc[k], next[k]);
            if (mine && k < n) {
                at(kGram, c, k) = acc[k];
                at(kP, c, k) = k == lane ? scale : 0.f;
                at(kQ, c, k) = 0.f;
            }
        }
    }

    for (int it = 0; it < a.admm_iter; ++it) {
        // The lower triangle from its rows' owners, mirrored: exactly symmetric.
#pragma unroll
        for (int k = 0; k < N; ++k) {
            if (k <= lane) g[lane * N + k] = next[k];
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < N; ++k) {
            if (k > lane) g[lane * N + k] = g[k * N + lane];
            next[k] = 0.f;
        }
        __syncthreads();
        if (it > 0) {
            // g = V^T g V with the eigenvectors of the step before, which the
            // sweeps then go on from.
#pragma unroll
            for (int k = 0; k < N; ++k) {
                float sum = 0.f;
#pragma unroll 1
                for (int j = 0; j < N; ++j) sum = fmaf(g[lane * N + j], v[j * N + k], sum);
                acc[k] = sum;
            }
#pragma unroll
            for (int k = 0; k < N; ++k) u[lane * N + k] = acc[k];
            __syncthreads();
#pragma unroll
            for (int k = 0; k < N; ++k) {
                float sum = 0.f;
#pragma unroll 1
                for (int i = 0; i < N; ++i) sum = fmaf(v[i * N + lane], u[i * N + k], sum);
                acc[k] = sum;
            }
#pragma unroll
            for (int k = 0; k < N; ++k) {
                if (k <= lane) g[lane * N + k] = acc[k];
            }
            __syncthreads();
#pragma unroll
            for (int k = 0; k < N; ++k) {
                if (k > lane) g[lane * N + k] = g[k * N + lane];
            }
            __syncthreads();
        }
        block_jacobi<float>(g, v, N, cs, &flag, it == 0, lane, N);
        // g = C with M = V C.
        if (lane == 0) kept = gram_shrink_spectrum<N>(g, v, rank, constant * (2.f / rho), start_k, true);
        __syncthreads();
        const bool last = it + 1 == a.admm_iter;
        const float next_rho = fminf(1e4f, a.mu * rho);
        const float inv_rho = 1.f / rho, inv_next = 1.f / next_rho, half_next = 0.5f * next_rho;
        for (int c = 0; c < 3; ++c) {
            const float w = weight[c];
            const float reciprocal = 1.f / (w + half_next);
            // Row `lane` of Z_c = (P_c + Q_c / rho) V C, then of Q_c', P_c'
            // and T_c' (into t); on the last iteration t keeps Z_c.
            float z[N];
#pragma unroll
            for (int k = 0; k < N; ++k) acc[k] = mine && k < n ? fmaf(at(kQ, c, k), inv_rho, at(kP, c, k)) : 0.f;
#pragma unroll
            for (int i = 0; i < N; ++i) {
                float sum = 0.f;
#pragma unroll
                for (int k = 0; k < N; ++k) sum = fmaf(acc[k], v[k * N + i], sum);
                z[i] = sum;
            }
#pragma unroll
            for (int j = 0; j < N; ++j) {
                float sum = 0.f;
#pragma unroll
                for (int i = 0; i < N; ++i) sum = fmaf(z[i], g[i * N + j], sum);
                acc[j] = sum;
            }
#pragma unroll
            for (int j = 0; j < N; ++j) {
                if (last) {
                    t[lane * N + j] = acc[j];
                    continue;
                }
                const bool inside = mine && j < n;
                const float p_old = inside ? at(kP, c, j) : 0.f, q_old = inside ? at(kQ, c, j) : 0.f;
                const float q_new = fmaf(rho, p_old - acc[j], q_old);
                const float p_new = ((lane == j ? w : 0.f) + half_next * (acc[j] - q_new * inv_next)) * reciprocal;
                if (inside) {
                    at(kP, c, j) = p_new;
                    at(kQ, c, j) = q_new;
                }
                t[lane * N + j] = inside ? fmaf(q_new, inv_next, p_new) : 0.f;
            }
            __syncthreads();
            if (last) {
                // Output column `lane` of Y_c Z_c.
                if (mine && (!fused || cell >= 0)) {
                    const float patch_weight = a.adaptive && kept > 0 ? 1.f / static_cast<float>(kept) : 1.f;
#pragma unroll
                    for (int k = 0; k < N; ++k) acc[k] = t[k * N + lane];
                    for (int p = 0; p < area; ++p) {
                        float y[N];
                        const float mean = load_row(c, p, y);
                        float sum = 0.f;
#pragma unroll
                        for (int k = 0; k < N; ++k) sum = fmaf(y[k], acc[k], sum);
                        if (fused) {
                            fixed_add(a.fused.num + c * a.fused.channel_step,
                                      cell + (p / a.block) * a.fused.pitch + p % a.block, patch_weight * (sum + mean));
                        } else {
                            out[c * channel_values + lane * area + p] = sum + mean;
                        }
                    }
                    if (fused) fixed_add(a.fused.den + c * a.fused.channel_step, cell, patch_weight);
                }
                __syncthreads();
                continue;
            }
            // Row `lane` of G_c T (into u), then of T^T (G_c T) (into next).
#pragma unroll
            for (int j = 0; j < N; ++j) acc[j] = 0.f;
#pragma unroll
            for (int k = 0; k < N; ++k) {
                const float gram = mine && k < n ? at(kGram, c, k) : 0.f;
#pragma unroll
                for (int j = 0; j < N; ++j) acc[j] = fmaf(gram, t[k * N + j], acc[j]);
            }
#pragma unroll
            for (int j = 0; j < N; ++j) u[lane * N + j] = acc[j];
            __syncthreads();
#pragma unroll
            for (int i = 0; i < N; ++i) {
                const float own = t[i * N + lane];
#pragma unroll
                for (int k = 0; k < N; ++k) next[k] = fmaf(own, u[i * N + k], next[k]);
            }
            __syncthreads();
        }
        rho = next_rho;
    }
    if (fused || !live) return;
    const float patch_weight = a.adaptive && kept > 0 ? 1.f / static_cast<float>(kept) : 1.f;
    for (int j = lane; j < a.group; j += N) {
        a.patches[static_cast<long long>(r) * a.group + j] =
            j < n ? AggregatePatch{match[j].x, match[j].y, match[j].t, patch_weight} : AggregatePatch{0, 0, -1, 0.f};
    }
}

}  // namespace

void mcwnnm_filter_groups(const McwnnmGroupArgs& args, cudaStream_t stream) {
    if (args.batch <= 0) return;
    const unsigned batch = static_cast<unsigned>(args.batch);
    if (args.group <= 8) {
        mcwnnm_group_kernel<8><<<(batch + 63) / 64, 64, 0, stream>>>(args);
    } else if (args.group <= 16) {
        mcwnnm_block_kernel<16><<<(batch + 1) / 2, 32, 0, stream>>>(args);
    } else {
        mcwnnm_block_kernel<32><<<batch, 32, 0, stream>>>(args);  // nss::kWnnmMaxGroup
    }
    NSS_CUDA_CHECK_LAUNCH();
}

}  // namespace nss_cuda
