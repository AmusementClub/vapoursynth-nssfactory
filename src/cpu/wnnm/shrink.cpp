#include "nss/cpu_api.hpp"
#include "nss/cpu_common.hpp"
#include "cpu/wnnm/jacobi8.hpp"

#include <algorithm>
#include <cmath>

namespace nss {

int wnnm_shrink(float* group, int m, int n, int lda, float sigma, int residual, int adaptive,
                float* adaptive_weight, float* work, int work_floats) {
    if (n <= 0 || m <= 0 || m > kSvdMaxM || n > kSvdMaxN) {
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

    const float constant = 8.f * std::sqrt(2.0f * static_cast<float>(n)) * sigma * sigma;
    const int k = sv_shrink(S, std::min(m, n), constant, residual ? 0 : 1);

    if (adaptive_weight) {
        if (adaptive) {
            *adaptive_weight = (k > 0) ? (1.f / static_cast<float>(k)) : 1.f;
        } else {
            *adaptive_weight = 1.f;
        }
    }

    if (m < n) {
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < m; ++j) {
                U[j + i * ldu] *= S[i];
            }
        }
    } else {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < k; ++j) {
                Vt[j + i * ldvt] *= S[j];
            }
        }
    }

    if (lda == m && ldu == m && ldvt == n) {
        gemm_nn_hwy(m, n, k, U, ldu, Vt, ldvt, group, lda);
    } else {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                float s = 0.f;
                for (int t = 0; t < k; ++t) {
                    s += U[i + t * ldu] * Vt[t + j * ldvt];
                }
                group[i + j * lda] = s;
            }
        }
    }

    if (residual) {
        group_center_add(group, m, n, lda, mean);
    }
    return 0;
}

}  // namespace nss
