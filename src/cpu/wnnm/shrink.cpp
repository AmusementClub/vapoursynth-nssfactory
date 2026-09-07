#include "nss/cpu_api.hpp"
#include "nss/cpu_common.hpp"
#include "cpu/wnnm/jacobi8.hpp"
#include "cpu/finishers.hpp"

#include <algorithm>
#include <cmath>

namespace nss {

int wnnm_shrink(float* group, int m, int n, int lda, float sigma, int residual, int adaptive,
                float* adaptive_weight, float* work, int work_floats) {
    if (!group || lda < m || n <= 0 || m <= 0 || m > kSvdMaxM || n > kSvdMaxN) {
        return -1;
    }
    const int ldu = m;
    const int ldvt = n;
    const int need = wnnm_shrink_work_floats(m, n);
    float* buf = work;
    int cap = work_floats;
    if (!buf || cap < need) {
        static thread_local float local[kSvdMaxM * kSvdMaxN + kSvdMaxN + kSvdMaxN * kSvdMaxN + kSvdMaxM +
                                        kSvdMaxM * kSvdMaxN * 6 + kSvdMaxN * kSvdMaxN * 8 + 256];
        buf = local;
        cap = static_cast<int>(sizeof(local) / sizeof(local[0]));
    }
    float* U = buf;
    float* S = U + m * n;
    float* Vt = S + n;
    float* mean = Vt + n * n;
    float* svd_work = mean + m;
    const int svd_cap = cap - (m * n + n + n * n + m);

    if (residual) {
        group_center_sub(group, m, n, lda, mean);
    }

    if (svd_economy(m, n, group, lda, U, ldu, S, Vt, ldvt, svd_work, svd_cap) != 0) {
        if (residual) {
            group_center_add(group, m, n, lda, mean);
        }
        return -1;
    }

    detail::finish_wnnm(group, m, n, lda, sigma, residual, adaptive, adaptive_weight, U, S, Vt, mean);
    return 0;
}

}  // namespace nss
