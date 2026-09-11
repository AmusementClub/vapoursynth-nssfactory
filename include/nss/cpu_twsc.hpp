#pragma once
#include "nss/checked.hpp"

namespace nss {

inline int pca_work_floats(int m, int n) {
    checked_solver_shape(m, n);
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

}  // namespace nss
