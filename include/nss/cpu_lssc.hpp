#pragma once

#include "nss/params.hpp"

#include <algorithm>
#include <cstddef>

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

// Workspace-only variant used by the frame pipeline. `scratch` holds k*m
// centroid floats; counts is reused as the k-element accumulator, so no heap
// allocation is needed in the hot path.
inline int lssc_cluster_work_floats(int m, int n, int nclusters) {
    const int mm = m < 1 ? 1 : m;
    const int nn = n < 1 ? 1 : n;
    const int kk = std::max(1, std::min(nn, nclusters < 1 ? 1 : nclusters));
    const int acc_floats = (kk * static_cast<int>(sizeof(int)) + static_cast<int>(sizeof(float)) - 1) /
                           static_cast<int>(sizeof(float));
    return kk * mm + acc_floats + 16;
}
void lssc_cluster_workspace(const float* patches, int m, int n, int lda, int nclusters, int* assign, int* counts,
                            float* scratch, int scratch_floats);

// ℓ1,2 row-soft: for each atom row of A (atoms × n, column-major), shrink the n-vector.
void lssc_group_soft(float* A, int atoms, int n, int lda_a, float lambda);

// Per-patch OMP. Used in K-SVD-lite only; denoise path is ISTA + group-soft, not SOMP.
int lssc_omp(const float* y, int m, const float* D, int atoms, int ldd, int sparsity, float* a);
inline int lssc_omp_work_floats(int m, int atoms, int sparsity) {
    const int mm = m < 1 ? 1 : m;
    const int aa = atoms < 1 ? 1 : atoms;
    const int kk = std::max(1, std::min(8, std::min(aa, sparsity < 1 ? 1 : sparsity)));
    const int support_f = (kk * static_cast<int>(sizeof(int)) + static_cast<int>(sizeof(float)) - 1) /
                          static_cast<int>(sizeof(float));
    // r, support, used, selected dictionary, Gram, rhs, correlations.
    return mm + support_f + aa + mm * kk + kk * kk + kk + aa + 32;
}
int lssc_omp_workspace(const float* y, int m, const float* D, int atoms, int ldd, int sparsity, float* a,
                       float* work, int work_floats);

// Overcomplete D (m × atoms, column-major). DCT-II basis + patch atoms; optional K-SVD-lite.
// Not Mairal ODL/SPAMS.
void lssc_dict_init(float* D, int m, int atoms, int ldd, const float* patches, int n, int lda, int block,
                    int ksvd_iters, unsigned seed);

// Scratch requirement for the optional K-SVD-lite initialization. The normal
// denoise path passes this region explicitly; the legacy wrapper may allocate
// it for standalone callers.
inline int lssc_dict_work_floats(int m, int atoms, int n, int ksvd_iters) {
    if (ksvd_iters <= 0) {
        return 16;
    }
    const int mm = m < 1 ? 1 : m;
    const int aa = atoms < 1 ? 1 : atoms;
    const int ns = std::min(std::max(n, 1), 256);
    const int nr = std::min(ns, kSvdMaxN);
    const int int_floats = (nr * static_cast<int>(sizeof(int)) + static_cast<int>(sizeof(float)) - 1) /
                           static_cast<int>(sizeof(float));
    // Y, A, aj, E, U, S, Vt, support, R, previous atom, previous coeffs,
    // and the caller-owned OMP scratch used by each K-SVD coding pass.
    return mm * ns + aa * ns + aa + mm * nr + mm * nr + nr + nr * nr + int_floats + mm * ns + mm + nr +
           lssc_omp_work_floats(mm, aa, 8) + 32;
}
void lssc_dict_init_workspace(float* D, int m, int atoms, int ldd, const float* patches, int n, int lda, int block,
                              int ksvd_iters, unsigned seed, float* work, int work_floats);

// Per-frame dictionary state. The context borrows all storage from the
// caller's workspace and must not outlive that frame.
struct LsscPreparedContext {
    const float* dictionary = nullptr;
    const float* transpose = nullptr;  // atoms x m, column-major
    int m = 0;
    int atoms = 0;
    int ldd = 0;
    float lipschitz = 1.f;
};

inline int lssc_prepare_work_floats(int m, int atoms) {
    const int mm = m < 1 ? 1 : m;
    const int aa = atoms < 1 ? 1 : atoms;
    // transpose[atoms*m] + power-vector[atoms] + image-vector[m]
    return aa * mm + aa + mm + 16;
}

// Build a frame-local transpose and Lipschitz estimate. The context borrows
// `D` and `work`; no input-dependent state may be retained across frames.
int lssc_prepare_context(const float* D, int m, int atoms, int ldd, float* work, int work_floats,
                         LsscPreparedContext* context);

inline int lssc_reconstruct_prepared_work_floats(int m, int n, int atoms) {
    const int mm = m < 1 ? 1 : m;
    const int nn = n < 1 ? 1 : n;
    const int aa = atoms < 1 ? 1 : atoms;
    // mean[n] + A[atoms*n] + R[m*n] + G[atoms*n]
    return nn + aa * nn + mm * nn + aa * nn + 16;
}

inline int lssc_reconstruct_work_floats(int m, int n, int atoms) {
    const int mm = m < 1 ? 1 : m;
    const int nn = n < 1 ? 1 : n;
    const int aa = atoms < 1 ? 1 : atoms;
    // Legacy path: prepared group workspace plus a local transpose and power
    // vectors. Prepared callers can use the smaller group-only contract.
    return lssc_reconstruct_prepared_work_floats(mm, nn, aa) + aa * mm + aa + mm + 16;
}

// ℓ_{1,2} ISTA: A ← prox_{μλ}(A + μ Dᵀ(Y − DA)), Y = DA. In-place on packed patches.
void lssc_reconstruct(float* patches, int m, int n, int lda, const float* D, int atoms, int ldd, float sigma,
                      float* work = nullptr, int work_floats = 0);

// Same reconstruction using a frame-local prepared dictionary context.
void lssc_reconstruct_prepared(float* patches, int m, int n, int lda, const LsscPreparedContext* context,
                               float sigma, float* work = nullptr, int work_floats = 0);

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
    const int offsets_f =
        ((ncl + 1) * static_cast<int>(sizeof(int)) + static_cast<int>(sizeof(float)) - 1) /
        static_cast<int>(sizeof(float));
    const int group_work = lssc_reconstruct_prepared_work_floats(m, np, atoms);
    const int prepare_work = lssc_prepare_work_floats(m, atoms);
    const int dict_work = lssc_dict_work_floats(m, atoms, np, 1);
    const int cluster_work = lssc_cluster_work_floats(m, np, ncl);
    // patches + D + assign + counts + prefix/cursors + members + group +
    // per-cluster ISTA + frame-local prepared dictionary.
    return np * m + m * atoms + assign_f + counts_f + offsets_f + counts_f + assign_f + cluster_work + np * m +
           dict_work + group_work + prepare_work;
}

// Extract grid patches, k-means cluster, ISTA group-sparse code, aggregate_add.
// Not ICCV 2009 LSSC (no Si=Sj / SOMP / ODL). work holds lssc_denoise_work_floats floats.
void lssc_denoise_plane(const float* src, int width, int height, int sstride, float* num, float* den, int buf_stride,
                        int block, int step, float sigma, float* work = nullptr, int work_floats = 0);

}  // namespace nss
