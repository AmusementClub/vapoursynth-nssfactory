#pragma once

namespace nss {

inline int twsc_pca_soft_work_floats(int m, int n) {
    const int local = m * n + n + n * n + m + m * n;
    const int svd = m * n * 6 + n * n * 8 + n + 256;
    return local + svd;
}

// Demean, SVD, B = UᵀY. group is left demeaned. U is m×r (ldu=m), S is r, B is r×n (ldb=r).
// On SVD failure restores the mean and returns -1. Otherwise returns r.
int pca_project(float* group, int m, int n, int lda, float* U, float* S, float* B, float* mean, float* work,
                int work_floats);

// group = U B + mean.
void pca_reconstruct(float* group, int m, int n, int lda, const float* U, const float* B, const float* mean);

// Demean, SVD, S = sqrt(max(s² − n σ0², 0)), C = soft(UᵀY, σ_j² / S).
// col_sigma[j] is the column noise (nullptr → sigma). col_w[j] = 1/(σ_j+ε) (nullable).
// row_w[i] is W1 on row i (nullptr → 1). Rows are scaled by W1 before PCA and unscaled after.
int twsc_pca_soft(float* group, int m, int n, int lda, float sigma, float* work, int work_floats,
                  const float* col_sigma = nullptr, float* col_w = nullptr, const float* row_w = nullptr);

}  // namespace nss
