#pragma once

#include "nss/params.hpp"
#include "nss/checked.hpp"

#include <algorithm>
#include <cstddef>

namespace nss {

inline int lssc_axis_count(int len, int block, int step) {
    if (len < block || block < 1 || step < 1) {
        return 0;
    }
    return checked_int((static_cast<std::uint64_t>(len - block) + step - 1) / step + 1);
}

inline int lssc_grid_count(int width, int height, int block, int step) {
    return checked_int(static_cast<std::uint64_t>(lssc_axis_count(width, block, step)) * lssc_axis_count(height, block, step));
}

// K-means / greedy assignment on packed patches (column-major). Every patch is assigned.
// assign[i] ∈ [0, nclusters). counts[c] = |{i : assign[i] = c}| (size nclusters).
void lssc_cluster(const float* patches, int m, int n, int lda, int nclusters, int* assign, int* counts);

// Workspace-only variant used by the frame pipeline. `scratch` holds k*m
// centroid floats; counts is reused as the k-element accumulator, so no heap
// allocation is needed in the hot path.
inline int lssc_cluster_work_floats(int m, int n, int nclusters) {
    const std::uint64_t mm = m < 1 ? 1 : m;
    const std::uint64_t nn = n < 1 ? 1 : n;
    const std::uint64_t kk = std::max<std::uint64_t>(1, std::min<std::uint64_t>(nn, nclusters < 1 ? 1 : nclusters));
    const int acc_floats = (kk * static_cast<int>(sizeof(int)) + static_cast<int>(sizeof(float)) - 1) /
                           static_cast<int>(sizeof(float));
    return checked_int(checked_sum(kk * mm, acc_floats, 16));
}
void lssc_cluster_workspace(const float* patches, int m, int n, int lda, int nclusters, int* assign, int* counts,
                            float* scratch, int scratch_floats);

// ℓ1,2 row-soft: for each atom row of A (atoms × n, column-major), shrink the n-vector.
void lssc_group_soft(float* A, int atoms, int n, int lda_a, float lambda);

// Per-patch OMP. Used in K-SVD-lite only; denoise path is ISTA + group-soft, not SOMP.
int lssc_omp(const float* y, int m, const float* D, int atoms, int ldd, int sparsity, float* a);
inline int lssc_omp_work_floats(int m, int atoms, int sparsity) {
    const std::uint64_t mm = m < 1 ? 1 : m;
    const std::uint64_t aa = atoms < 1 ? 1 : atoms;
    const std::uint64_t kk = std::max<std::uint64_t>(1, std::min<std::uint64_t>(8, std::min<std::uint64_t>(aa, sparsity < 1 ? 1 : sparsity)));
    const int support_f = (kk * static_cast<int>(sizeof(int)) + static_cast<int>(sizeof(float)) - 1) /
                          static_cast<int>(sizeof(float));
    // r, support, used, selected dictionary, Gram, rhs, correlations.
    return checked_int(checked_sum(mm, support_f, aa, mm * kk, kk * kk, kk, aa, 32));
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
        return checked_int(16);
    }
    const std::uint64_t mm = m < 1 ? 1 : m;
    const std::uint64_t aa = atoms < 1 ? 1 : atoms;
    const std::uint64_t ns = std::min(std::max(n, 1), 256);
    const std::uint64_t nr = std::min<std::uint64_t>(ns, kSvdMaxN);
    const int int_floats = (nr * static_cast<int>(sizeof(int)) + static_cast<int>(sizeof(float)) - 1) /
                           static_cast<int>(sizeof(float));
    // Y, A, aj, E, U, S, Vt, support, R, previous atom, previous coeffs,
    // and the caller-owned OMP scratch used by each K-SVD coding pass.
    return checked_int(checked_sum(mm * ns, aa * ns, aa, mm * nr, mm * nr, nr, nr * nr, int_floats, mm * ns, mm, nr,
           lssc_omp_work_floats(mm, aa, 8), 32));
}
void lssc_dict_init_workspace(float* D, int m, int atoms, int ldd, const float* patches, int n, int lda, int block,
                              int ksvd_iters, unsigned seed, float* work, int work_floats, bool avx2_gemm = false);

// Per-frame dictionary state. The context borrows all storage from the
// caller's workspace and must not outlive that frame.
struct LsscPreparedContext {
    const float* dictionary = nullptr;
    const float* transpose = nullptr;  // atoms x m, column-major
    int m = 0;
    int atoms = 0;
    int ldd = 0;
    float lipschitz = 1.f;
    bool avx2_gemm = false;
};

inline int lssc_prepare_work_floats(int m, int atoms) {
    const std::uint64_t mm = m < 1 ? 1 : m;
    const std::uint64_t aa = atoms < 1 ? 1 : atoms;
    // transpose[atoms*m] + power-vector[atoms] + image-vector[m]
    return checked_int(checked_sum(aa * mm, aa, mm, 16));
}

// Build a frame-local transpose and Lipschitz estimate. The context borrows
// `D` and `work`; no input-dependent state may be retained across frames.
int lssc_prepare_context(const float* D, int m, int atoms, int ldd, float* work, int work_floats,
                         LsscPreparedContext* context);

inline int lssc_reconstruct_prepared_work_floats(int m, int n, int atoms) {
    const std::uint64_t mm = m < 1 ? 1 : m;
    const std::uint64_t nn = n < 1 ? 1 : n;
    const std::uint64_t aa = atoms < 1 ? 1 : atoms;
    // mean[n] + A[atoms*n] + R[m*n] + G[atoms*n]
    return checked_int(checked_sum(nn, aa * nn, mm * nn, aa * nn, 16));
}

inline int lssc_reconstruct_work_floats(int m, int n, int atoms) {
    const std::uint64_t mm = m < 1 ? 1 : m;
    const std::uint64_t nn = n < 1 ? 1 : n;
    const std::uint64_t aa = atoms < 1 ? 1 : atoms;
    // Legacy path: prepared group workspace plus a local transpose and power
    // vectors. Prepared callers can use the smaller group-only contract.
    return checked_int(checked_sum(lssc_reconstruct_prepared_work_floats(mm, nn, aa), aa * mm, aa, mm, 16));
}

// ℓ_{1,2} ISTA: A ← prox_{μλ}(A + μ Dᵀ(Y − DA)), Y = DA. In-place on packed patches.
void lssc_reconstruct(float* patches, int m, int n, int lda, const float* D, int atoms, int ldd, float sigma,
                      float* work = nullptr, int work_floats = 0);

// Same reconstruction using a frame-local prepared dictionary context.
void lssc_reconstruct_prepared(float* patches, int m, int n, int lda, const LsscPreparedContext* context,
                               float sigma, float* work = nullptr, int work_floats = 0);

inline int lssc_denoise_work_floats(int width, int height, int block, int step) {
    const int np0 = lssc_grid_count(width, height, block, step);
    const std::uint64_t np = np0 < 1 ? 1 : np0;
    const std::uint64_t b = block < 1 ? 1 : block;
    const int m = checked_int(b * b);
    const std::uint64_t atoms = np < kLsscDefaultAtoms ? np : kLsscDefaultAtoms;
    const std::uint64_t ncl = np < kLsscDefaultClusters ? np : kLsscDefaultClusters;
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
    return checked_int(checked_sum(np * m, m * atoms, assign_f, counts_f, offsets_f, counts_f, assign_f, cluster_work, np * m,
           dict_work, group_work, prepare_work));
}

// Extract grid patches, k-means cluster, ISTA group-sparse code, aggregate_add.
// Not ICCV 2009 LSSC (no Si=Sj / SOMP / ODL). work holds lssc_denoise_work_floats floats.
void lssc_denoise_plane(const float* src, int width, int height, int sstride, float* num, float* den, int buf_stride,
                        int block, int step, float sigma, float* work = nullptr, int work_floats = 0);

}  // namespace nss
