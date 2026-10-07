// SPDX-License-Identifier: GPL-2.0-only
#include "cuda/mcwnnm/kernels.hpp"
#include "cuda/common/block_jacobi.cuh"
#include "cuda/common/fixed_accumulate.cuh"
#include "cuda/common/gram_shrink.cuh"
#include "cuda/runtime/error.hpp"

namespace nss_cuda {
namespace {

// Scratch of the kernels for up to 8 columns: matrix m of channel c at
// (m * 3 + c) * group^2 (kVectors: channel 0 only), the aggregation weight at
// kMcwnnmWeight * group^2. Interleaved over chunks of 64 groups (element e of
// group r at e * width + r % 64, width the groups of the chunk), so the
// warps of the solve kernel and the 8-thread groups of the other two both
// touch consecutive words.
enum { kGram = 0, kP = 1, kQ = 2, kVectors = 3 };
constexpr int kMcwnnmWeight = 10, kMcwnnmChunk = 64, kMcwnnmLanes = 8, kMcwnnmWarps = 4;

__device__ __forceinline__ int mcwnnm_width(const McwnnmGroupArgs& a, int r) {
    return min(kMcwnnmChunk, a.batch - r / kMcwnnmChunk * kMcwnnmChunk);
}
__device__ __forceinline__ float* mcwnnm_state(const McwnnmGroupArgs& a, int r) {
    const long long per = 10LL * a.group * a.group + 1;  // mcwnnm_scratch_floats
    return a.scratch + (r / kMcwnnmChunk) * kMcwnnmChunk * per + r % kMcwnnmChunk;
}

// The centered pixel at `offset` of the group's patches and its mean.
__device__ __forceinline__ float mcwnnm_pixel(const float* const* col, long long offset, int n, bool residual,
                                              float* value) {
    constexpr int N = 8;
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < N; ++j) {
        value[j] = j < n ? col[j][offset] : 0.f;
        sum += value[j];
    }
    const float mean = residual ? sum * (1.f / static_cast<float>(n)) : 0.f;
#pragma unroll
    for (int j = 0; j < N; ++j) value[j] = j < n ? value[j] - mean : 0.f;
    return mean;
}

// Up to 8 columns, first of three kernels: the channel Gram matrices of the
// centered input. 8 threads per group, thread l taking the pixels of column
// l (l + 8, ...) of every patch, so a row of a patch is one request of the
// group's threads; a shuffle tree adds the 8 partial sums.
__global__ void mcwnnm_rows_kernel(McwnnmGroupArgs a) {
    constexpr int N = 8;
    const int lane = threadIdx.x % kMcwnnmLanes;
    const int slot = (blockIdx.x * blockDim.x + threadIdx.x) / kMcwnnmLanes;
    const bool live = slot < a.batch;
    const int r = live ? slot : a.batch - 1;  // idle groups mirror the last one and store nothing
    const int n = min(a.counts[r], a.group);
    const DeviceMatch* match = a.matches + static_cast<long long>(r) * a.group;
    const float* col[N];
#pragma unroll
    for (int j = 0; j < N; ++j) {
        const DeviceMatch& mj = match[j < n ? j : 0];
        col[j] = a.src[mj.t] + static_cast<long long>(mj.y) * a.pitch + mj.x;
    }
    const int square = a.group * a.group, width = mcwnnm_width(a, r);
    float* const state = mcwnnm_state(a, r);
    for (int cls = 0; cls < a.classes; ++cls) {
        // The class's channels into one sum.
        float g[N * (N + 1) / 2];
#pragma unroll
        for (int e = 0; e < N * (N + 1) / 2; ++e) g[e] = 0.f;
        for (int c = 0; c < 3; ++c) {
            if (a.class_of[c] != cls) continue;
            for (int row = 0; row < a.block; ++row) {
                for (int x = lane; x < a.block; x += kMcwnnmLanes) {
                    float y[N];
                    mcwnnm_pixel(col, c * a.channel_step + static_cast<long long>(row) * a.pitch + x, n, a.residual, y);
#pragma unroll
                    for (int j = 0; j < N; ++j) {
#pragma unroll
                        for (int k = 0; k <= j; ++k) {
                            g[j * (j + 1) / 2 + k] = fmaf(y[j], y[k], g[j * (j + 1) / 2 + k]);
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int j = 0; j < N; ++j) {
#pragma unroll
            for (int k = 0; k <= j; ++k) {
                const int e = j * (j + 1) / 2 + k;
                float sum = g[e];
#pragma unroll
                for (int offset = 1; offset < kMcwnnmLanes; offset <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, offset);
                if (live && j < n && e % kMcwnnmLanes == lane) {
                    state[static_cast<long long>((kGram * 3 + cls) * square + j * a.group + k) * width] = sum;
                    state[static_cast<long long>((kGram * 3 + cls) * square + k * a.group + j) * width] = sum;
                }
            }
        }
    }
}

// Third: the output Y_c Z_c in the layout of the first, with Z_c and the
// aggregation weight from the solve kernel.
__global__ void mcwnnm_apply_kernel(McwnnmGroupArgs a) {
    constexpr int N = 8;
    const int lane = threadIdx.x % kMcwnnmLanes;
    const int r = (blockIdx.x * blockDim.x + threadIdx.x) / kMcwnnmLanes;
    if (r >= a.batch) return;  // no shuffles or barriers below
    const int area = a.block * a.block;
    const int n = min(a.counts[r], a.group);
    const DeviceMatch* match = a.matches + static_cast<long long>(r) * a.group;
    const float* col[N];
#pragma unroll
    for (int j = 0; j < N; ++j) {
        const DeviceMatch& mj = match[j < n ? j : 0];
        col[j] = a.src[mj.t] + static_cast<long long>(mj.y) * a.pitch + mj.x;
    }
    const int square = a.group * a.group, width = mcwnnm_width(a, r);
    const float* const state = mcwnnm_state(a, r);
    const float patch_weight = state[static_cast<long long>(kMcwnnmWeight) * square * width];
    const bool fused = a.fused.num != nullptr;
    long long cell[N];
#pragma unroll
    for (int j = 0; j < N; ++j) {
        cell[j] = fused && j < n ? fixed_patch_cell(a.fused, match[j].t, match[j].y, match[j].x) : -1;
    }
    const long long channel_values = static_cast<long long>(a.batch) * a.group * area;
    float* const out = a.values + static_cast<long long>(r) * a.group * area;  // channel c at + c * channel_values
    for (int c = 0; c < 3; ++c) {
        float z[N * N];
#pragma unroll
        for (int k = 0; k < N; ++k) {
#pragma unroll
            for (int j = 0; j < N; ++j) {
                z[k * N + j] =
                    k < n && j < n ? state[static_cast<long long>((kP * 3 + a.class_of[c]) * square + k * a.group + j) * width]
                                   : 0.f;
            }
        }
        unsigned long long* const num = a.fused.num + c * a.fused.channel_step;
        for (int row = 0; row < a.block; ++row) {
            for (int x = lane; x < a.block; x += kMcwnnmLanes) {
                float y[N];
                const float mean =
                    mcwnnm_pixel(col, c * a.channel_step + static_cast<long long>(row) * a.pitch + x, n, a.residual, y);
                const int target = row * a.fused.pitch + x;
#pragma unroll
                for (int j = 0; j < N; ++j) {
                    float sum = 0.f;
#pragma unroll
                    for (int k = 0; k < N; ++k) sum = fmaf(y[k], z[k * N + j], sum);
                    if (fused) {
                        if (cell[j] >= 0) fixed_add(num, cell[j] + target, patch_weight * (sum + mean));
                    } else if (j < n) {
                        out[c * channel_values + j * area + row * a.block + x] = sum + mean;
                    }
                }
            }
        }
        if (fused && lane == 0) {
            // The den once per patch, at its first cell: the finish box-sums it.
#pragma unroll
            for (int j = 0; j < N; ++j) {
                if (cell[j] >= 0) fixed_add(a.fused.den + c * a.fused.channel_step, cell[j], patch_weight);
            }
        }
    }
    if (fused || lane != 0) return;
    for (int j = 0; j < a.group; ++j) {
        a.patches[static_cast<long long>(r) * a.group + j] =
            j < n ? AggregatePatch{match[j].x, match[j].y, match[j].t, patch_weight} : AggregatePatch{0, 0, -1, 0.f};
    }
}

// Second: the ADMM, one thread per group with fixed loop bounds and order
// (run-to-run identical), on the Gram matrices of the first kernel. N is the
// column capacity; columns beyond the group's count are zero and contribute
// zero singular values.
//
// The ADMM runs on n x n matrices instead of the m x n iterates. Every step
// after the eigen step treats a row of X and A alone and linearly, with
// coefficients that depend only on the row's channel, so row i of channel c
// stays x_i = y_i P_c, a_i = y_i Q_c for the centered input row y_i. Then
//   Z_c = (P_c + Q_c / rho) M,
//   Q_c' = Q_c + rho (P_c - Z_c),
//   P_c' = (w_c I + rho' / 2 (Z_c - Q_c' / rho')) / (w_c + rho' / 2),
// and the Gram matrix of the next shrinkage input T_c' = P_c' + Q_c' / rho'
// is sum_c T_c'^T (Y_c^T Y_c) T_c'. The image rows are streamed twice, by
// the kernels around this one: for the three channel Gram matrices, and for
// the output Y_c Z_c of the last iteration.
//
// Channels of equal sigma have equal weights, hence equal P, Q and Z, and
// their terms of the next Gram matrix add up to T^T (sum of their Gram
// matrices) T. They iterate as one class (McwnnmGroupArgs::classes): with
// one sigma for all three, a third of the state and of the work on it.
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
    // Per channel the input Gram matrix (from the rows kernel), P and Q, and
    // the eigenvectors of the last eigen step, each group x group and
    // interleaved across the block's groups (mcwnnm_state).
    const int lanes = mcwnnm_width(a, r);
    const int square = a.group * a.group;
    float* const state = mcwnnm_state(a, r);
    // A row's address, then its element: written this way the kernel keeps
    // the 8 strides k * lanes in registers instead of an offset per element,
    // which pushed the matrices out to local memory (9.9 M spill requests
    // per launch against 1.3 M; 148 fps against 169).
    const long long stride = lanes;
    const auto at = [&](int matrix, int c, int j, int k) -> float& {
        float* const row = state + static_cast<long long>((matrix * 3 + c) * square + j * a.group) * stride;
        return row[k * stride];
    };

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

    // P_c = scale_c I and Q_c = 0 (X0 = argmin with Z = A = 0), whose
    // shrinkage input is X0 itself.
    float rho = a.rho;
#pragma unroll
    for (int i = 0; i < N * N; ++i) next[i] = 0.f;
    for (int c = 0; c < a.classes; ++c) {  // c: a class of channels from here on
        const float scale = weight[a.channel_of[c]] / (weight[a.channel_of[c]] + 0.5f * rho);
#pragma unroll
        for (int j = 0; j < N; ++j) {
#pragma unroll
            for (int k = 0; k <= j; ++k) {
                const float value = j < n ? at(kGram, c, j, k) : 0.f;
                next[j * N + k] = fmaf(scale * scale, value, next[j * N + k]);
                if (j < n) {
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
        for (int c = 0; c < a.classes; ++c) {
            const float w = weight[a.channel_of[c]];
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
                // Z_c for the apply kernel, in P_c's place.
#pragma unroll
                for (int row = 0; row < N; ++row) {
#pragma unroll
                    for (int j = 0; j < N; ++j) {
                        if (row < n && j < n) at(kP, c, row, j) = v[row * N + j];
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
    state[static_cast<long long>(kMcwnnmWeight) * square * lanes] = a.adaptive && kept > 0 ? 1.f / static_cast<float>(kept) : 1.f;
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
    __shared__ float g_all[kGroups][N * N], v_all[kGroups][N * N], t_all[kGroups][N * N];
    __shared__ float cs_all[kGroups][N + 2];
    // u (products that all threads read) is a fourth matrix only with
    // several classes: with one, g is free whenever u is in use, and the
    // matrix less is a third more groups per SM.
    extern __shared__ float u_all[];
    __shared__ int flag, kept_all[kGroups];
    const int slot = threadIdx.x / N, lane = threadIdx.x % N;
    float* const g = g_all[slot];
    float* const v = v_all[slot];
    float* const t = t_all[slot];
    float* const u = a.classes == 1 ? g : u_all + slot * N * N;
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
    for (int c = 0; c < a.classes; ++c) {  // c: a class of channels from here on (see the thread kernel)
#pragma unroll
        for (int k = 0; k < N; ++k) acc[k] = 0.f;
        for (int channel = 0; channel < 3; ++channel) {
            if (a.class_of[channel] != c) continue;
            for (int p = 0; p < area; ++p) {
                float y[N];
                load_row(channel, p, y);
                const float own = y[lane];
#pragma unroll
                for (int k = 0; k < N; ++k) acc[k] = fmaf(own, y[k], acc[k]);
            }
        }
        const float scale = weight[a.channel_of[c]] / (weight[a.channel_of[c]] + 0.5f * rho);
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
            __syncthreads();  // u may be g: every thread has read it before one writes
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
        for (int c = 0; c < a.classes; ++c) {
            const float w = weight[a.channel_of[c]];
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
                // Output column `lane` of Y Z for every channel of the class.
                if (mine && (!fused || cell >= 0)) {
                    const float patch_weight = a.adaptive && kept > 0 ? 1.f / static_cast<float>(kept) : 1.f;
#pragma unroll
                    for (int k = 0; k < N; ++k) acc[k] = t[k * N + lane];
                    for (int channel = 0; channel < 3; ++channel) {
                        if (a.class_of[channel] != c) continue;
                        for (int p = 0; p < area; ++p) {
                            float y[N];
                            const float mean = load_row(channel, p, y);
                            float sum = 0.f;
#pragma unroll
                            for (int k = 0; k < N; ++k) sum = fmaf(y[k], acc[k], sum);
                            if (fused) {
                                fixed_add(a.fused.num + channel * a.fused.channel_step,
                                          cell + (p / a.block) * a.fused.pitch + p % a.block,
                                          patch_weight * (sum + mean));
                            } else {
                                out[channel * channel_values + lane * area + p] = sum + mean;
                            }
                        }
                        if (fused) fixed_add(a.fused.den + channel * a.fused.channel_step, cell, patch_weight);
                    }
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

void mcwnnm_filter_groups(const McwnnmGroupArgs& given, cudaStream_t stream) {
    if (given.batch <= 0) return;
    McwnnmGroupArgs args = given;
    args.classes = 0;
    for (int c = 0; c < 3; ++c) {
        int cls = 0;
        // A channel without noise is pinned to its input by a weight of 1e12
        // and stays a class of its own: two of them as one class agreed with
        // the CPU to 102 dB where they reach 136 dB apart.
        while (cls < args.classes && !(args.sigma[c] > 0.f && args.sigma[args.channel_of[cls]] == args.sigma[c])) ++cls;
        if (cls == args.classes) args.channel_of[args.classes++] = c;
        args.class_of[c] = cls;
    }
    const unsigned batch = static_cast<unsigned>(args.batch);
    if (args.group <= 8) {
        const unsigned per_block = kMcwnnmWarps * 32 / kMcwnnmLanes;
        mcwnnm_rows_kernel<<<(batch + per_block - 1) / per_block, kMcwnnmWarps * 32, 0, stream>>>(args);
        NSS_CUDA_CHECK_LAUNCH();
        mcwnnm_group_kernel<8><<<(batch + kMcwnnmChunk - 1) / kMcwnnmChunk, kMcwnnmChunk, 0, stream>>>(args);
        NSS_CUDA_CHECK_LAUNCH();
        mcwnnm_apply_kernel<<<(batch + per_block - 1) / per_block, kMcwnnmWarps * 32, 0, stream>>>(args);
    } else if (args.group <= 16) {
        mcwnnm_block_kernel<16><<<(batch + 1) / 2, 32, args.classes == 1 ? 0 : 2 * 16 * 16 * sizeof(float), stream>>>(args);
    } else {
        // nss::kWnnmMaxGroup
        mcwnnm_block_kernel<32><<<batch, 32, args.classes == 1 ? 0 : 32 * 32 * sizeof(float), stream>>>(args);
    }
    NSS_CUDA_CHECK_LAUNCH();
}

}  // namespace nss_cuda
