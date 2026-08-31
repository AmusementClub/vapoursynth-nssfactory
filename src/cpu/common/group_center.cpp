#include "nss/cpu_common.hpp"
#include "cpu/hwy_config.hpp"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/common/group_center.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

void GroupCenterSub(float* group, int m, int n, int lda, float* mean) {
    if (!group || !mean || m < 1 || n < 1 || lda < m) {
        return;
    }
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const auto inv = hn::Set(d, 1.f / static_cast<float>(n));
    int i = 0;
    for (; i + N <= m; i += N) {
        auto acc = hn::Zero(d);
        for (int j = 0; j < n; ++j) {
            acc = hn::Add(acc, hn::LoadU(d, group + i + j * lda));
        }
        const auto mu = hn::Mul(acc, inv);
        hn::StoreU(mu, d, mean + i);
        for (int j = 0; j < n; ++j) {
            hn::StoreU(hn::Sub(hn::LoadU(d, group + i + j * lda), mu), d, group + i + j * lda);
        }
    }
    for (; i < m; ++i) {
        float s = 0.f;
        for (int j = 0; j < n; ++j) {
            s += group[i + j * lda];
        }
        mean[i] = s / static_cast<float>(n);
        for (int j = 0; j < n; ++j) {
            group[i + j * lda] -= mean[i];
        }
    }
}

void GroupCenterAdd(float* group, int m, int n, int lda, const float* mean) {
    if (!group || !mean || m < 1 || n < 1 || lda < m) {
        return;
    }
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    for (int j = 0; j < n; ++j) {
        int i = 0;
        float* col = group + j * lda;
        for (; i + N <= m; i += N) {
            hn::StoreU(hn::Add(hn::LoadU(d, col + i), hn::LoadU(d, mean + i)), d, col + i);
        }
        for (; i < m; ++i) {
            col[i] += mean[i];
        }
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(GroupCenterSub);
HWY_EXPORT(GroupCenterAdd);

void group_center_sub(float* group, int m, int n, int lda, float* mean) {
    HWY_DYNAMIC_DISPATCH(GroupCenterSub)(group, m, n, lda, mean);
}

void group_center_add(float* group, int m, int n, int lda, const float* mean) {
    HWY_DYNAMIC_DISPATCH(GroupCenterAdd)(group, m, n, lda, mean);
}

}  // namespace nss
#endif
