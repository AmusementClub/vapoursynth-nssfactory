// SPDX-License-Identifier: GPL-2.0-only
// Weighted nuclear-norm shrinkage of a small Gram matrix, shared by the WNNM
// family (nss::sv_shrink through A^T A = V S^2 V^T): for a group matrix A with
// Gram matrix g, returns M = V diag(s'/s) V^T so that the shrunk
// reconstruction is A M. Singular values walk the ordered spectrum: the first
// `start` are kept as they are, the following ones shrink to
// (s + sqrt(s^2 - constant)) / 2 until one falls to the threshold.
#pragma once

#include <cuda_runtime.h>

namespace nss_cuda {

// One Jacobi rotation of the symmetric g in the (p, q) plane, also applied to
// the columns of v. Branch-free: without an off-diagonal element it is the
// identity.
template <int N>
__device__ __forceinline__ void jacobi_rotate(float* g, float* v, int p, int q) {
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

// Cyclic Jacobi sweeps on the N x N symmetric matrix g (full storage), with
// every rotation also applied to the columns of v: on return the diagonal of
// g holds the eigenvalues. Fixed operation order (deterministic).
// Up to N = 8 the sweep is unrolled over its pairs: with p and q constant,
// g and v stay in registers. (An unroll count that is an expression is not
// honoured: the pairs then stay a loop and both matrices live in local
// memory, where the kernel waits on every element.)
template <int N>
__device__ __forceinline__ void jacobi_sweeps(float* g, float* v) {
    for (int sweep = 0; sweep < 16; ++sweep) {
        float off = 0.f, diag = 0.f;
#pragma unroll
        for (int i = 0; i < N; ++i) {
            diag = fmaf(g[i * N + i], g[i * N + i], diag);
#pragma unroll
            for (int j = i + 1; j < N; ++j) off = fmaf(g[i * N + j], g[i * N + j], off);
        }
        if (off <= 1e-13f * diag || off <= 1e-30f) break;
        if constexpr (N <= 8) {
#pragma unroll
            for (int p = 0; p < N - 1; ++p) {
#pragma unroll
                for (int q = p + 1; q < N; ++q) jacobi_rotate<N>(g, v, p, q);
            }
        } else {
            for (int p = 0; p < N - 1; ++p) {
                for (int q = p + 1; q < N; ++q) jacobi_rotate<N>(g, v, p, q);
            }
        }
    }
}

// Eigendecomposition of g: as jacobi_sweeps from v = I, so that column i of
// v is the eigenvector of eigenvalue i.
template <int N>
__device__ __forceinline__ void jacobi_eigen(float* g, float* v) {
#pragma unroll
    for (int i = 0; i < N * N; ++i) v[i] = (i / N == i % N) ? 1.f : 0.f;
    jacobi_sweeps<N>(g, v);
}

// Singular values s = sqrt(max(eigenvalue, 0)) and their indices in
// descending order (stable).
template <int N>
__device__ __forceinline__ void singular_order(const float* g, float* s, int* order) {
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
}

// gram_shrink. g: N x N symmetric Gram matrix (full storage), replaced by M. v: N x N
// workspace. rank: number of singular values that can be nonzero. Returns the
// number of kept singular values.
// The part of gram_shrink after the eigendecomposition: g carries the
// eigenvalues on its diagonal and v the eigenvectors. With `codes`, g returns
// C = diag(gain) V^T instead of M (M = V C; a block kernel forms that product
// across its threads).
template <int N>
__device__ __forceinline__ int gram_shrink_spectrum(float* g, const float* v, int rank, float constant, int start_k,
                                                    bool codes = false) {
    float s[N];
    int order[N];
    singular_order<N>(g, s, order);
    const int start = min(start_k, rank);
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
    if (codes) {
        for (int k = 0; k < N; ++k) {
            for (int j = 0; j < N; ++j) g[k * N + j] = gain[k] * v[j * N + k];
        }
        return kept;
    }
    // M = V diag(gain) V^T.
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
    return kept;
}

template <int N>
__device__ __forceinline__ int gram_shrink(float* g, float* v, int rank, float constant, int start_k) {
    jacobi_eigen<N>(g, v);
    return gram_shrink_spectrum<N>(g, v, rank, constant, start_k);
}

}  // namespace nss_cuda
