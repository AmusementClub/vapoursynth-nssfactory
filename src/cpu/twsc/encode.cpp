#include "nss/cpu_twsc.hpp"
#include "nss/cpu_api.hpp"
#include "nss/cpu_common.hpp"
#include "cpu/hwy_config.hpp"
#include "cpu/wnnm/jacobi8.hpp"
#include "cpu/finishers.hpp"

#include <algorithm>
#include <cmath>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/twsc/encode.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

int PcaProject(float* group, int m, int n, int lda, float* U, float* S, float* B, float* mean, float* work,
               int work_floats) {
    if (!group || !U || !S || !B || !mean || n <= 0 || m <= 0 || lda < m || m > kSvdMaxM || n > kSvdMaxN) {
        return -1;
    }
    const int r = std::min(m, n);
    const int svd_need = m * n * 6 + n * n * 8 + n + 256;
    if (!work || work_floats < n * n + svd_need) {
        return -1;
    }
    float* Vt = work;
    float* svd_work = Vt + n * n;
    const int svd_cap = work_floats - n * n;
    group_center_sub(group, m, n, lda, mean);
    if (svd_economy(m, n, group, lda, U, m, S, Vt, n, svd_work, svd_cap) != 0) {
        group_center_add(group, m, n, lda, mean);
        return -1;
    }
    gemm_tn_hwy(m, n, r, U, m, group, lda, B, r);
    return r;
}

void PcaReconstruct(float* group, int m, int n, int lda, const float* U, const float* B, const float* mean) {
    if (!group || !U || !B || !mean || m < 1 || n < 1 || lda < m) {
        return;
    }
    detail::finish_pca_reconstruction(group, m, n, lda, U, B, mean);
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(PcaProject);
HWY_EXPORT(PcaReconstruct);

int pca_project(float* group, int m, int n, int lda, float* U, float* S, float* B, float* mean, float* work,
                int work_floats) {
    return HWY_DYNAMIC_DISPATCH(PcaProject)(group, m, n, lda, U, S, B, mean, work, work_floats);
}

void pca_reconstruct(float* group, int m, int n, int lda, const float* U, const float* B, const float* mean) {
    HWY_DYNAMIC_DISPATCH(PcaReconstruct)(group, m, n, lda, U, B, mean);
}

}  // namespace nss
#endif
