// SPDX-License-Identifier: GPL-2.0-only
#pragma once

#include "nss/params.hpp"
#include "nss/checked.hpp"
#include "nss/params/lssc.hpp"

#include <algorithm>
#include <cstddef>

namespace nss {

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

// Per-patch OMP. Used in K-SVD-lite only; denoise path is FISTA + group-soft, not SOMP.
int lssc_omp(const float* y, int m, const float* D, int atoms, int ldd, int sparsity, float* a);
inline int lssc_omp_work_floats(int m, int atoms, int sparsity) {
    const std::uint64_t mm = m < 1 ? 1 : m;
    const std::uint64_t aa = atoms < 1 ? 1 : atoms;
    const std::uint64_t kk = std::max<std::uint64_t>(1, std::min<std::uint64_t>(8, std::min<std::uint64_t>(aa, sparsity < 1 ? 1 : sparsity)));
    const int support_f = (kk * static_cast<int>(sizeof(int)) + static_cast<int>(sizeof(float)) - 1) /
                          static_cast<int>(sizeof(float));
    // Retain the conservative legacy bound for callers and frame accounting.
    // Stable OMP stores a double residual in 2*m float slots plus an atoms-byte
    // used mask; even sparsity=1 reserves at least that much storage here.
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
    const float* transpose = nullptr;  // atoms x m, column-major: D^T
    // Coding (params/lssc.hpp): gradient matrix P = D^T M (layout of
    // transpose), per-atom penalty factors ||M d_a||, and the step bound L'.
    const float* gradient = nullptr;
    const float* lambda_scale = nullptr;
    int m = 0;
    int atoms = 0;
    int ldd = 0;
    float lipschitz = 1.f;         // power estimate of ||D||^2
    float coding_lipschitz = 1.f;  // L' of the preconditioned problem
    bool avx2_gemm = false;
};

inline int lssc_prepare_work_floats(int m, int atoms) {
    const std::uint64_t mm = m < 1 ? 1 : m;
    const std::uint64_t aa = atoms < 1 ? 1 : atoms;
    // transpose[atoms*m] + power-vector[atoms] + image-vector[m] +
    // gradient[atoms*m] + lambda_scale[atoms] + Cholesky factor[m*m]
    return checked_int(checked_sum(aa * mm, aa, mm, aa * mm, aa, mm * mm, 16));
}

// Build the frame-local transpose, Lipschitz estimate and preconditioned
// coding operator (gradient, lambda_scale, coding_lipschitz; without a usable
// preconditioner they fall back to D^T, 1 and lipschitz). The context borrows
// `D` and `work`; no input-dependent state may be retained across frames.
int lssc_prepare_context(const float* D, int m, int atoms, int ldd, float* work, int work_floats,
                         LsscPreparedContext* context);

// Model version 4 coding (params/lssc.hpp). The dictionary borrows D and
// `storage` (lssc_omp_dictionary_floats): its transpose (pixel-major), Gram
// matrix and inverse atom norms.
struct LsscOmpDictionary {
    const float* dictionary = nullptr;  // m x atoms, column-major, leading dimension ldd
    const float* transpose = nullptr;   // m rows of atoms
    const float* gram = nullptr;        // atoms x atoms
    const float* inv_norm = nullptr;    // atoms
    int m = 0;
    int atoms = 0;
    int ldd = 0;
};
inline int lssc_omp_cap(int m, int atoms) { return std::max(0, std::min({kLsscOmpMaxAtoms, m, atoms})); }
inline int lssc_omp_dictionary_floats(int m, int atoms) {
    const std::uint64_t mm = m < 1 ? 1 : m;
    const std::uint64_t aa = atoms < 1 ? 1 : atoms;
    return checked_int(checked_sum(aa * mm, aa * aa, aa, 16));
}
inline int lssc_omp_code_work_floats(int m, int atoms) {
    const std::uint64_t mm = m < 1 ? 1 : m;
    const std::uint64_t aa = atoms < 1 ? 1 : atoms;
    return checked_int(checked_sum(3 * aa, mm, 16));
}
void lssc_omp_prepare(const float* D, int m, int atoms, int ldd, float* storage, LsscOmpDictionary* out);
// Codes one patch (m pixels) into rec (m pixels); returns the support size.
// work holds lssc_omp_code_work_floats floats.
int lssc_omp_code(const float* patch, const LsscOmpDictionary& dict, float sigma, float* rec, float* work);
// The weight of the noisy plane in the average (sigma normalized to [0, 1]).
float lssc_image_weight(int block, int step, float sigma);

// Optional matrix-kernel capability, independent of the Highway vector target.
bool lssc_sme_compiled() noexcept;
bool lssc_sme_available() noexcept;
int lssc_gemm_pack_work_floats(int m, int n, int atoms) noexcept;

inline int lssc_reconstruct_prepared_work_floats(int m, int n, int atoms) {
    const std::uint64_t mm = m < 1 ? 1 : m;
    const std::uint64_t nn = n < 1 ? 1 : n;
    const std::uint64_t aa = atoms < 1 ? 1 : atoms;
    // mean[n] + A[atoms*n] + R[m*n] + G[atoms*n] + previous iterate[atoms*n] +
    // optional bounded GEMM pack.
    const int pack = lssc_gemm_pack_work_floats(static_cast<int>(mm), static_cast<int>(nn), static_cast<int>(aa));
    return checked_int(checked_sum(nn, aa * nn, mm * nn, aa * nn, aa * nn, pack, 16));
}

inline int lssc_reconstruct_work_floats(int m, int n, int atoms) {
    const std::uint64_t mm = m < 1 ? 1 : m;
    const std::uint64_t nn = n < 1 ? 1 : n;
    const std::uint64_t aa = atoms < 1 ? 1 : atoms;
    // Legacy path: prepared group workspace plus a local prepared context.
    // Prepared callers can use the smaller group-only contract.
    return checked_int(checked_sum(lssc_reconstruct_prepared_work_floats(mm, nn, aa),
                                   lssc_prepare_work_floats(static_cast<int>(mm), static_cast<int>(aa)), 16));
}

// ℓ_{1,2} group sparse coding of one cluster (params/lssc.hpp): preconditioned
// FISTA, A ← prox_{μλ}(Z + μ P(Y − DZ)), output DA + mean. In-place on packed patches.
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
    const int atoms = static_cast<int>(np < kLsscDefaultAtoms ? np : kLsscDefaultAtoms);
    // patches + D + coding dictionary + K-SVD scratch + coding scratch + one
    // reconstructed patch.
    return checked_int(checked_sum(np * m, static_cast<std::uint64_t>(m) * atoms, lssc_omp_dictionary_floats(m, atoms),
                                   lssc_dict_work_floats(m, atoms, static_cast<int>(np), 1),
                                   lssc_omp_code_work_floats(m, atoms), m, 32));
}

// Model version 4 (params/lssc.hpp): grid patches, the plane's dictionary,
// each patch coded by error-constrained OMP and aggregated, plus the noisy
// plane at lssc_image_weight. Not ICCV 2009 LSSC (no simultaneous coding, no
// ODL). work holds lssc_denoise_work_floats floats.
void lssc_denoise_plane(const float* src, int width, int height, int sstride, float* num, float* den, int buf_stride,
                        int block, int step, float sigma, float* work = nullptr, int work_floats = 0);

}  // namespace nss
