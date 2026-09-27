// SPDX-License-Identifier: GPL-2.0-only
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
    int j = 0;
#if !NSS_ALIGNMENT_GENERIC
    // Four independent column chains fill FMA latency. Each output element
    // keeps its own serial l-order MulAdd chain, bit-identical to the
    // single-column loop below (mirrors the WNNM/NCSR SvdChunk precedent).
    for (; j + 4 <= n; j += 4) {
        int i = 0;
        for (; i + lanes <= m; i += lanes) {
            auto s0 = hn::Zero(d), s1 = hn::Zero(d), s2 = hn::Zero(d), s3 = hn::Zero(d);
            for (int l = 0; l < k; ++l) {
                const auto av = hn::LoadU(d, a + i + l * m);
                s0 = hn::MulAdd(av, hn::Set(d, b[l + (j + 0) * k]), s0);
                s1 = hn::MulAdd(av, hn::Set(d, b[l + (j + 1) * k]), s1);
                s2 = hn::MulAdd(av, hn::Set(d, b[l + (j + 2) * k]), s2);
                s3 = hn::MulAdd(av, hn::Set(d, b[l + (j + 3) * k]), s3);
            }
            hn::StoreU(s0, d, c + i + (j + 0) * m);
            hn::StoreU(s1, d, c + i + (j + 1) * m);
            hn::StoreU(s2, d, c + i + (j + 2) * m);
            hn::StoreU(s3, d, c + i + (j + 3) * m);
        }
        for (; i < m; ++i) {
            double sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
            for (int l = 0; l < k; ++l) {
                sum0 += a[i + l * m] * b[l + (j + 0) * k];
                sum1 += a[i + l * m] * b[l + (j + 1) * k];
                sum2 += a[i + l * m] * b[l + (j + 2) * k];
                sum3 += a[i + l * m] * b[l + (j + 3) * k];
            }
            c[i + (j + 0) * m] = sum0;
            c[i + (j + 1) * m] = sum1;
            c[i + (j + 2) * m] = sum2;
            c[i + (j + 3) * m] = sum3;
        }
    }
#endif
    for (; j < n; ++j) {
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
    for (int j = 0; j < n; ++j) {
        int i = 0;
#if !NSS_ALIGNMENT_GENERIC
        // Four independent dot-product chains share the B column load. Each
        // output keeps its lane-strided MulAdd order, single ReduceSum and
        // scalar tail start, bit-identical to the single-dot loop below.
        for (; i + 4 <= k; i += 4) {
            auto v0 = hn::Zero(d), v1 = hn::Zero(d), v2 = hn::Zero(d), v3 = hn::Zero(d);
            int l = 0;
            for (; l + lanes <= m; l += lanes) {
                const auto bv = hn::LoadU(d, b + l + j * m);
                v0 = hn::MulAdd(hn::LoadU(d, a + l + (i + 0) * m), bv, v0);
                v1 = hn::MulAdd(hn::LoadU(d, a + l + (i + 1) * m), bv, v1);
                v2 = hn::MulAdd(hn::LoadU(d, a + l + (i + 2) * m), bv, v2);
                v3 = hn::MulAdd(hn::LoadU(d, a + l + (i + 3) * m), bv, v3);
            }
            double sum0 = hn::ReduceSum(d, v0);
            double sum1 = hn::ReduceSum(d, v1);
            double sum2 = hn::ReduceSum(d, v2);
            double sum3 = hn::ReduceSum(d, v3);
            for (; l < m; ++l) {
                sum0 += a[l + (i + 0) * m] * b[l + j * m];
                sum1 += a[l + (i + 1) * m] * b[l + j * m];
                sum2 += a[l + (i + 2) * m] * b[l + j * m];
                sum3 += a[l + (i + 3) * m] * b[l + j * m];
            }
            c[(i + 0) + j * k] = sum0;
            c[(i + 1) + j * k] = sum1;
            c[(i + 2) + j * k] = sum2;
            c[(i + 3) + j * k] = sum3;
        }
#endif
        for (; i < k; ++i) {
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
