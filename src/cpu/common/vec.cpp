#include "nss/cpu_common.hpp"
#include "cpu/hwy_config.hpp"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/common/vec.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

float DotN(const float* a, const float* b, int n) {
    if (!a || !b || n < 1) {
        return 0.f;
    }
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    auto acc = hn::Zero(d);
    int i = 0;
    for (; i + N <= n; i += N) {
        acc = hn::MulAdd(hn::LoadU(d, a + i), hn::LoadU(d, b + i), acc);
    }
    float s = hn::ReduceSum(d, acc);
    for (; i < n; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

float SsdVec(const float* a, const float* b, int n) {
    if (!a || !b || n < 1) {
        return 0.f;
    }
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    auto acc = hn::Zero(d);
    int i = 0;
    for (; i + N <= n; i += N) {
        const auto e = hn::Sub(hn::LoadU(d, a + i), hn::LoadU(d, b + i));
        acc = hn::MulAdd(e, e, acc);
    }
    float s = hn::ReduceSum(d, acc);
    for (; i < n; ++i) {
        const float e = a[i] - b[i];
        s += e * e;
    }
    return s;
}

void AxpyN(float* y, const float* x, float a, int n) {
    if (!y || !x || n < 1) {
        return;
    }
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const auto va = hn::Set(d, a);
    int i = 0;
    for (; i + N <= n; i += N) {
        hn::StoreU(hn::MulAdd(va, hn::LoadU(d, x + i), hn::LoadU(d, y + i)), d, y + i);
    }
    for (; i < n; ++i) {
        y[i] += a * x[i];
    }
}

void ScaleN(float* y, float s, int n) {
    if (!y || n < 1) {
        return;
    }
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const auto vs = hn::Set(d, s);
    int i = 0;
    for (; i + N <= n; i += N) {
        hn::StoreU(hn::Mul(vs, hn::LoadU(d, y + i)), d, y + i);
    }
    for (; i < n; ++i) {
        y[i] *= s;
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(DotN);
HWY_EXPORT(SsdVec);
HWY_EXPORT(AxpyN);
HWY_EXPORT(ScaleN);

float dot_n(const float* a, const float* b, int n) {
    return HWY_DYNAMIC_DISPATCH(DotN)(a, b, n);
}

float ssd_vec(const float* a, const float* b, int n) {
    return HWY_DYNAMIC_DISPATCH(SsdVec)(a, b, n);
}

void axpy_n(float* y, const float* x, float a, int n) {
    HWY_DYNAMIC_DISPATCH(AxpyN)(y, x, a, n);
}

void scale_n(float* y, float s, int n) {
    HWY_DYNAMIC_DISPATCH(ScaleN)(y, s, n);
}

}  // namespace nss
#endif
