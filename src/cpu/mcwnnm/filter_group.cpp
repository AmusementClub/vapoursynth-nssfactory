#include "nss/cpu_mcwnnm.hpp"

#include "nss/cpu_common.hpp"
#include "nss/params.hpp"

#include <cmath>

namespace nss {

int mcwnnm_filter_group(float* Y, int m, int n, int lda, int nch, const float* sigma, int admm_iter, float rho,
                        float mu, int residual, int adaptive, float* adaptive_weight, float* work, int work_floats) {
    if (!Y || m < 1 || n < 1 || lda < m || m > kSvdMaxM || n > kSvdMaxN) {
        return -1;
    }
    float mean_tls[kSvdMaxM];
    float* mean = mean_tls;
    const int admm_need = mcwnnm_admm_work_floats(m, n);
    if (residual) {
        if (work && work_floats >= admm_need + m) {
            mean = work + admm_need;
        } else if (m > kSvdMaxM) {
            return -1;
        }
        group_center_sub(Y, m, n, lda, mean);
    }
    const int kept = mcwnnm_admm(Y, m, n, lda, nch, sigma, admm_iter, rho, mu, residual ? 0 : 1, work, work_floats);
    if (residual) {
        group_center_add(Y, m, n, lda, mean);
    }
    if (kept < 0) {
        return -1;
    }
    bool any_sigma = false;
    if (sigma) {
        for (int c = 0; c < nch; ++c) {
            if (sigma[c] > 0.f && is_finite_bits(sigma[c])) {
                any_sigma = true;
                break;
            }
        }
    }
    if (adaptive_weight) {
        if (adaptive && any_sigma && kept > 0) {
            *adaptive_weight = 1.f / static_cast<float>(kept);
        } else {
            *adaptive_weight = 1.f;
        }
    }
    return 0;
}

}  // namespace nss
