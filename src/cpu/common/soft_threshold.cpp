#include "nss/cpu_common.hpp"
#include "cpu/hwy_config.hpp"

#include <cmath>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/common/soft_threshold.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

void SoftThreshold(float* x, int n, float tau) {
    if (!x || n < 1) {
        return;
    }
    const float t = tau < 0.f ? 0.f : tau;
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const auto vt = hn::Set(d, t);
    const auto z = hn::Zero(d);
    int i = 0;
    for (; i + N <= n; i += N) {
        const auto v = hn::LoadU(d, x + i);
        const auto a = hn::Abs(v);
        const auto shrunk = hn::CopySign(hn::Sub(a, vt), v);
        hn::StoreU(hn::IfThenElse(hn::Gt(a, vt), shrunk, z), d, x + i);
    }
    for (; i < n; ++i) {
        const float v = x[i];
        const float a = std::fabs(v);
        x[i] = (a > t) ? std::copysign(a - t, v) : 0.f;
    }
}

void SoftThresholdVar(float* x, const float* tau, int n) {
    if (!x || !tau || n < 1) {
        return;
    }
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const auto z = hn::Zero(d);
    int i = 0;
    for (; i + N <= n; i += N) {
        const auto v = hn::LoadU(d, x + i);
        const auto vt = hn::Max(hn::LoadU(d, tau + i), z);
        const auto a = hn::Abs(v);
        const auto shrunk = hn::CopySign(hn::Sub(a, vt), v);
        hn::StoreU(hn::IfThenElse(hn::Gt(a, vt), shrunk, z), d, x + i);
    }
    for (; i < n; ++i) {
        const float t = tau[i] < 0.f ? 0.f : tau[i];
        const float v = x[i];
        const float a = std::fabs(v);
        x[i] = (a > t) ? std::copysign(a - t, v) : 0.f;
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(SoftThreshold);
HWY_EXPORT(SoftThresholdVar);

void soft_threshold(float* x, int n, float tau) {
    HWY_DYNAMIC_DISPATCH(SoftThreshold)(x, n, tau);
}

void soft_threshold_var(float* x, const float* tau, int n) {
    HWY_DYNAMIC_DISPATCH(SoftThresholdVar)(x, tau, n);
}

}  // namespace nss
#endif
