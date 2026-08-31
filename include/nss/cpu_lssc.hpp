#pragma once

#include "nss/params.hpp"

namespace nss {

inline int lssc_axis_count(int len, int block, int step) {
    if (len < block || block < 1 || step < 1) {
        return 0;
    }
    int n = 0;
    for (int x = 0; x < len - block + step; x += step) {
        ++n;
    }
    return n;
}

inline int lssc_grid_count(int width, int height, int block, int step) {
    return lssc_axis_count(width, block, step) * lssc_axis_count(height, block, step);
}

// K-means / greedy assignment on packed patches (column-major). Every patch is assigned.
// assign[i] ∈ [0, nclusters). counts[c] = |{i : assign[i] = c}| (size nclusters).
void lssc_cluster(const float* patches, int m, int n, int lda, int nclusters, int* assign, int* counts);

// Overcomplete D (m × atoms, column-major). DCT-II basis + patch atoms; optional K-SVD-lite.
// Not Mairal ODL/SPAMS.
void lssc_dict_init(float* D, int m, int atoms, int ldd, const float* patches, int n, int lda, int block,
                    int ksvd_iters, unsigned seed);

// ℓ1,2 row-soft: for each atom row of A (atoms × n, column-major), shrink the n-vector.
void lssc_group_soft(float* A, int atoms, int n, int lda_a, float lambda);

// Per-patch OMP. Used in K-SVD-lite only; denoise path is ISTA + group-soft, not SOMP.
int lssc_omp(const float* y, int m, const float* D, int atoms, int ldd, int sparsity, float* a);

inline int lssc_reconstruct_work_floats(int m, int n, int atoms) {
    const int mm = m < 1 ? 1 : m;
    const int nn = n < 1 ? 1 : n;
    const int aa = atoms < 1 ? 1 : atoms;
    // mean[n] + A[atoms*n] + R[m*n] + G[atoms*n] + DT[atoms*m]
    return nn + aa * nn + mm * nn + aa * nn + aa * mm + 16;
}

// ℓ_{1,2} ISTA: A ← prox_{μλ}(A + μ Dᵀ(Y − DA)), Y = DA. In-place on packed patches.
void lssc_reconstruct(float* patches, int m, int n, int lda, const float* D, int atoms, int ldd, float sigma,
                      float* work = nullptr, int work_floats = 0);

inline int lssc_denoise_work_floats(int width, int height, int block, int step) {
    const int np0 = lssc_grid_count(width, height, block, step);
    const int np = np0 < 1 ? 1 : np0;
    const int b = block < 1 ? 1 : block;
    const int m = b * b;
    const int atoms = np < kLsscDefaultAtoms ? np : kLsscDefaultAtoms;
    const int ncl = np < kLsscDefaultClusters ? np : kLsscDefaultClusters;
    const int assign_f =
        (np * static_cast<int>(sizeof(int)) + static_cast<int>(sizeof(float)) - 1) / static_cast<int>(sizeof(float));
    const int counts_f =
        (ncl * static_cast<int>(sizeof(int)) + static_cast<int>(sizeof(float)) - 1) / static_cast<int>(sizeof(float));
    // patches + D + assign + counts + members + group + ISTA (worst cluster = np)
    return np * m + m * atoms + assign_f + counts_f + assign_f + np * m + lssc_reconstruct_work_floats(m, np, atoms);
}

// Extract grid patches, k-means cluster, ISTA group-sparse code, aggregate_add.
// Not ICCV 2009 LSSC (no Si=Sj / SOMP / ODL). work holds lssc_denoise_work_floats floats.
void lssc_denoise_plane(const float* src, int width, int height, int sstride, float* num, float* den, int buf_stride,
                        int block, int step, float sigma, float* work = nullptr, int work_floats = 0);

}  // namespace nss
