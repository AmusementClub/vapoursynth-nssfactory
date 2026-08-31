#pragma once

namespace nss {

void jacobi_svd_8(const float* A, int lda, float* U, int ldu, float* S, float* Vt, int ldvt, float* V);
int householder_qr_hwy(int m, int n, const float* A, int lda, float* Q, int ldq, float* R, int ldr, float* W, float* Vh,
                       float* beta);
void gemm_nn_hwy(int m, int n, int k, const float* A, int lda, const float* B, int ldb, float* C, int ldc);
// C = Aᵀ B. A is m×k, B is m×n, C is k×n.
void gemm_tn_hwy(int m, int n, int k, const float* A, int lda, const float* B, int ldb, float* C, int ldc);

}  // namespace nss
