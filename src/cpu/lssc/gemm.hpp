#pragma once

namespace nss {

// Column-major B becomes row-major packed B. Input and output are disjoint.
void lssc_pack_columns_hwy(const float* input, int rows, int columns, int stride, float* packed);

// Same overwrite product as gemm_nn_hwy. No allocation; false reports Highway
// fallback. Inputs, output and the caller-owned packing workspace are disjoint.
bool lssc_gemm_nn(int m, int n, int k, const float* a, int lda,
                  const float* b, int ldb, float* c, int ldc,
                  float* scratch, int scratch_floats, bool avx2_gemm = false);

}  // namespace nss
