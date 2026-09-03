#pragma once

#include "nss/cpu_batch.hpp"

namespace nss {

void jacobi_svd_8(const float* A, int lda, float* U, int ldu, float* S, float* Vt, int ldvt, float* V);
int householder_qr_hwy(int m, int n, const float* A, int lda, float* Q, int ldq, float* R, int ldr, float* W, float* Vh,
                       float* beta, bool form_q = true);
void apply_householder_hwy(float* matrix, int ld, int ncols, const float* v, int len, float beta);
void gemm_nn_hwy(int m, int n, int k, const float* A, int lda, const float* B, int ldb, float* C, int ldc);
// C = Aᵀ B. A is m×k, B is m×n, C is k×n.
void gemm_tn_hwy(int m, int n, int k, const float* A, int lda, const float* B, int ldb, float* C, int ldc);

// Runs homogeneous m x 8 economy SVDs with independent matrices mapped to
// Highway lanes. Pointer arrays contain count entries and may use different
// leading dimensions. The caller retains ownership of every buffer.
int svd_economy_8_batch_hwy(int m, const float* const* A, const int* lda, float* const* U, const int* ldu,
                            float* const* S, float* const* Vt, const int* ldvt, int count);
// U/S-only variant for PCA consumers that never read the right singular vectors.
int svd_economy_8_batch_u_hwy(int m, const float* const* A, const int* lda, float* const* U, const int* ldu,
                              float* const* S, int count);
#if defined(__GNUC__) || defined(__clang__)
__attribute__((visibility("hidden")))
#endif
int svd_economy_8_batch_qreplay_hwy(int m, const float* const* A, const int* lda, float* const* U, const int* ldu,
                                    float* const* S, float* const* Vt, const int* ldvt, int count);
#if defined(__GNUC__) || defined(__clang__)
__attribute__((visibility("hidden")))
#endif
int svd_economy_8_batch_u_qreplay_hwy(int m, const float* const* A, const int* lda, float* const* U, const int* ldu,
                                      float* const* S, int count);
#if defined(__GNUC__) || defined(__clang__)
__attribute__((visibility("hidden")))
#endif
int svd_economy_8_batch_u_compat_hwy(int m, const float* const* A, const int* lda, float* const* U, const int* ldu,
                                     float* const* S, float* const* Vt, const int* ldvt, int count);

}  // namespace nss
