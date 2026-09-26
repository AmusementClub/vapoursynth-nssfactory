#include "nss/cpu_nlh_full.hpp"
#include "cpu/hwy_config.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/nlh/fused.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// Fused Basic/Wiener shrink over the q*n Haar coefficient matrix of one
// source row. Elementwise FP64: every coefficient keeps the scalar loop's
// exact IEEE operation sequence (promote, mul, add, div, iterated mul,
// demote), so values are bit-identical; this TU is pinned no-fast-math /
// no-contract for that reason. The structural mask (j > 0 && k >= max(0,
// q-2)) is recomputed from the flat index; q is a power of two by contract.
void NlhShrinkFull(float* matrix, const float* reference, int q, int n, double threshold, double noise,
                   int wiener_iterations, bool wiener) {
    const int total = q * n;
    const hn::ScalableTag<double> dd;
    const hn::Rebind<float, decltype(dd)> df;
    const hn::RebindToSigned<decltype(dd)> di;
    const int lanes = static_cast<int>(hn::Lanes(dd));
    const int structural_k = std::max(0, q - 2);
    const auto vthr = hn::Set(dd, threshold);
    int index = 0;
    for (; index + lanes <= total; index += lanes) {
        const auto idx = hn::Add(hn::Set(di, index), hn::Iota(di, 0));
        const auto jrow = hn::Gt(idx, hn::Set(di, q - 1));
        const auto kcol = hn::And(idx, hn::Set(di, q - 1));
        const auto structural = hn::And(jrow, hn::Ge(kcol, hn::Set(di, structural_k)));
        if (wiener) {
            const auto r = hn::PromoteTo(dd, hn::LoadU(df, reference + index));
            const auto r2 = hn::Mul(r, r);
            // The zero-noise limit is identity, including 0/0 coefficients.
            const auto gain = noise == 0.0 ? hn::Set(dd, 1.0) : hn::Div(r2, hn::Add(r2, hn::Set(dd, noise)));
            auto value = hn::PromoteTo(dd, hn::LoadU(df, matrix + index));
            for (int it = 0; it < wiener_iterations; ++it) value = hn::Mul(value, gain);
            hn::StoreU(hn::DemoteTo(df, value), df, matrix + index);
        } else {
            const auto value = hn::PromoteTo(dd, hn::LoadU(df, matrix + index));
            const auto small = hn::Lt(hn::Abs(value), vthr);
            const auto kill = hn::Or(small, hn::RebindMask(dd, structural));
            hn::StoreU(hn::DemoteTo(df, hn::IfThenElse(kill, hn::Zero(dd), value)), df, matrix + index);
        }
    }
    for (; index < total; ++index) {
        const int j = index / q;
        const int k = index - j * q;
        if (wiener) {
            const double r = reference[index];
            const double r2 = r * r;
            const double gain = noise == 0.0 ? 1.0 : r2 / (r2 + noise);
            double value = matrix[index];
            for (int it = 0; it < wiener_iterations; ++it) value *= gain;
            matrix[index] = static_cast<float>(value);
        } else if (std::abs(static_cast<double>(matrix[index])) < threshold ||
                   (j > 0 && k >= structural_k)) {
            matrix[index] = 0.f;
        }
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(NlhShrinkFull);

void nlh_shrink_full(float* matrix, const float* reference, int q, int n, double threshold, double noise,
                     int wiener_iterations, bool wiener) {
    HWY_DYNAMIC_DISPATCH(NlhShrinkFull)(matrix, reference, q, n, threshold, noise, wiener_iterations, wiener);
}

}  // namespace nss
#endif
