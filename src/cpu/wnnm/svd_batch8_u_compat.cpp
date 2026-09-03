#include "cpu/wnnm/jacobi8.hpp"

namespace nss {

int svd_economy_8_batch_u_compat_hwy(int m, const float* const* A, const int* lda, float* const* U, const int* ldu,
                                     float* const* S, float* const* Vt, const int* ldvt, int count) {
    (void)Vt;
    (void)ldvt;
    if (count < 1) {
        return svd_economy_8_batch_u_hwy(m, A, lda, U, ldu, S, count);
    }
    return svd_economy_8_batch_u_qreplay_hwy(m, A, lda, U, ldu, S, count);
}

}  // namespace nss
