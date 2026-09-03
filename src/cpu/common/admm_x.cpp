#include "nss/cpu_common.hpp"
#include "cpu/hwy_config.hpp"

#include <algorithm>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/common/admm_x.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

float ChannelWeightDiag(float* w2, int m, int nch, const float* sigma) {
    if (!w2 || !sigma || m < 1 || nch < 1 || m % nch != 0) {
        return 0.f;
    }
    const int area = m / nch;
    float smin = 0.f;
    for (int c = 0; c < nch; ++c) {
        if (sigma[c] > 0.f && is_finite_bits(sigma[c])) {
            smin = (smin == 0.f) ? sigma[c] : std::min(smin, sigma[c]);
        }
    }
    if (!(smin > 0.f)) {
        return 0.f;
    }
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    for (int c = 0; c < nch; ++c) {
        const float sc = (sigma[c] > 0.f && is_finite_bits(sigma[c])) ? sigma[c] : smin * 1e-6f;
        const float wc2 = (smin / sc) * (smin / sc);
        const auto v = hn::Set(d, wc2);
        float* row = w2 + c * area;
        int i = 0;
        for (; i + N <= area; i += N) {
            hn::StoreU(v, d, row + i);
        }
        for (; i < area; ++i) {
            row[i] = wc2;
        }
    }
    return smin;
}

void AdmmWeightedX(float* X, const float* Y, int ldy, const float* Z, const float* A, const float* w2, int m, int n,
                   float rho) {
    if (!X || !Y || !Z || !A || !w2 || m < 1 || n < 1 || ldy < m || !(rho > 0.f) || !is_finite_bits(rho)) {
        return;
    }
    const float rho2 = 0.5f * rho;
    const float inv_rho = 1.f / rho;
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const auto vrho2 = hn::Set(d, rho2);
    const auto vinv = hn::Set(d, inv_rho);
    const auto one = hn::Set(d, 1.f);
    int i = 0;
    for (; i + N <= m; i += N) {
        const auto w = hn::LoadU(d, w2 + i);
        const auto reciprocal = hn::Div(one, hn::Add(w, vrho2));
        for (int j = 0; j < n; ++j) {
            const float* ycol = Y + j * ldy;
            const float* zcol = Z + j * m;
            const float* acol = A + j * m;
            float* xcol = X + j * m;
            const auto yv = hn::LoadU(d, ycol + i);
            const auto zv = hn::LoadU(d, zcol + i);
            const auto av = hn::LoadU(d, acol + i);
            const auto num = hn::MulAdd(w, yv, hn::Mul(vrho2, hn::Sub(zv, hn::Mul(av, vinv))));
            hn::StoreU(hn::Mul(num, reciprocal), d, xcol + i);
        }
    }
    for (; i < m; ++i) {
        const float reciprocal = 1.f / (w2[i] + rho2);
        for (int j = 0; j < n; ++j) {
            const float* ycol = Y + j * ldy;
            const float* zcol = Z + j * m;
            const float* acol = A + j * m;
            float* xcol = X + j * m;
            xcol[i] = (w2[i] * ycol[i] + rho2 * (zcol[i] - acol[i] * inv_rho)) * reciprocal;
        }
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(ChannelWeightDiag);
HWY_EXPORT(AdmmWeightedX);

float channel_weight_diag(float* w2, int m, int nch, const float* sigma) {
    return HWY_DYNAMIC_DISPATCH(ChannelWeightDiag)(w2, m, nch, sigma);
}

void admm_weighted_x(float* X, const float* Y, int ldy, const float* Z, const float* A, const float* w2, int m, int n,
                     float rho) {
    HWY_DYNAMIC_DISPATCH(AdmmWeightedX)(X, Y, ldy, Z, A, w2, m, n, rho);
}

}  // namespace nss
#endif
