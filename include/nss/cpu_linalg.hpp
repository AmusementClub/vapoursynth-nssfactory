#pragma once

namespace nss {
// Row-major C[m,n] = A[m,k] * B[k,n]. Each product is rounded to
// float, then added in increasing k order to a double accumulator.
// Independent output columns are Highway lanes; no horizontal reduction,
// reassociation, or fused float multiplication/addition is used.
// Returns -1 for invalid views, 0 for success. Output must not alias inputs.
int gemm_f32_f64(const float* a,int lda,const float* b,int ldb,float* c,int ldc,
                 int m,int n,int k);
}
