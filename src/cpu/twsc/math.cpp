#include "nss/cpu_twsc_full.hpp"
#include "cpu/hwy_config.hpp"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/twsc/math.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;
void TwscGemmNN(int m, int n, int k, const double* a, const double* b, double* c) {
    const hn::ScalableTag<double> d;
    const int lanes = int(hn::Lanes(d));
    for (int j = 0; j < n; ++j) {
        int i = 0;
#if !NSS_ALIGNMENT_GENERIC
        for (; i + lanes <= m; i += lanes) {
            auto sum = hn::Zero(d);
            for (int l = 0; l < k; ++l) sum = hn::MulAdd(hn::LoadU(d, a + i + l * m), hn::Set(d, b[l + j * k]), sum);
            hn::StoreU(sum, d, c + i + j * m);
        }
#endif
        for (; i < m; ++i) {
            double sum = 0;
            for (int l = 0; l < k; ++l) sum += a[i + l * m] * b[l + j * k];
            c[i + j * m] = sum;
        }
    }
}
void TwscGemmTN(int m, int n, int k, const double* a, const double* b, double* c) {
    const hn::ScalableTag<double> d;
    const int lanes = int(hn::Lanes(d));
    for (int j = 0; j < n; ++j) for (int i = 0; i < k; ++i) {
        int l = 0;
        double sum = 0;
#if !NSS_ALIGNMENT_GENERIC
        auto v = hn::Zero(d);
        for (; l + lanes <= m; l += lanes) v = hn::MulAdd(hn::LoadU(d, a + l + i * m), hn::LoadU(d, b + l + j * m), v);
        sum = hn::ReduceSum(d, v);
#endif
        for (; l < m; ++l) sum += a[l + i * m] * b[l + j * m];
        c[i + j * k] = sum;
    }
}
} // namespace HWY_NAMESPACE
} // namespace nss
HWY_AFTER_NAMESPACE();
#if HWY_ONCE
namespace nss {
HWY_EXPORT(TwscGemmNN);
HWY_EXPORT(TwscGemmTN);
void twsc_gemm_nn(int m, int n, int k, const double* a, const double* b, double* c) {
    HWY_DYNAMIC_DISPATCH(TwscGemmNN)(m, n, k, a, b, c);
}
void twsc_gemm_tn(int m, int n, int k, const double* a, const double* b, double* c) {
    HWY_DYNAMIC_DISPATCH(TwscGemmTN)(m, n, k, a, b, c);
}
} // namespace nss
#endif
