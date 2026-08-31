#include "nss/cpu_common.hpp"
#include "cpu/hwy_config.hpp"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/common/iter_regularize.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

void IterRegularize(float* x, const float* y, int n, float delta) {
    if (!x || !y || n < 1) {
        return;
    }
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const auto vd = hn::Set(d, delta);
    int i = 0;
    for (; i + N <= n; i += N) {
        const auto xv = hn::LoadU(d, x + i);
        const auto yv = hn::LoadU(d, y + i);
        hn::StoreU(hn::MulAdd(vd, hn::Sub(yv, xv), xv), d, x + i);
    }
    for (; i < n; ++i) {
        x[i] += delta * (y[i] - x[i]);
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(IterRegularize);

void iter_regularize(float* x, const float* y, int n, float delta) {
    HWY_DYNAMIC_DISPATCH(IterRegularize)(x, y, n, delta);
}

}  // namespace nss
#endif
