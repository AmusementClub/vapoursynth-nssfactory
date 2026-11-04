// SPDX-License-Identifier: GPL-2.0-only
// The candidate mask of the image-level matcher's scan (image_ssd_row.hpp).
// FINITE_HOT class: the mask must see infinite and NaN distances, and with
// -ffinite-math-only clang folds the exponent test of a bit-cast float to
// false (AVX3, 2026-11-04: a non-finite distance late in a full row was
// skipped instead of reported).
#include "cpu/common/image_ssd_row.hpp"
#include "cpu/hwy_config.hpp"

#include <algorithm>
#include <cstdint>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/common/image_candidates.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

void ImageRowCandidates(const float* dist, int count, float worst, unsigned long long* mask) {
    const hn::ScalableTag<float> d;
    const hn::RebindToSigned<decltype(d)> di;
    const int N = static_cast<int>(hn::Lanes(d));
    const auto bound = hn::Set(d, worst);
    const auto exponent = hn::Set(di, 0x7f800000);
    HWY_ALIGN std::uint8_t bits[HWY_MAX_BYTES / 8 + 8];
    for (int w = 0; w < (count + 63) / 64; ++w) mask[w] = 0;
    for (int i = 0; i < count; i += N) {
        const size_t n = static_cast<size_t>(std::min(N, count - i));
        const auto v = hn::LoadN(d, dist + i, n);
        const auto special = hn::RebindMask(d, hn::Eq(hn::And(hn::BitCast(di, v), exponent), exponent));
        const auto take = hn::And(hn::Or(hn::Le(v, bound), special), hn::FirstN(d, n));
        hn::StoreMaskBits(d, take, bits);
        unsigned long long word = 0;
        for (int b = 0; b < (N + 7) / 8; ++b) word |= static_cast<unsigned long long>(bits[b]) << (8 * b);
        if (N < 64) word &= (1ull << N) - 1;
        mask[i / 64] |= word << (i % 64);
    }
}

ImageRowCandidatesKernel ResolveImageRowCandidates() { return &ImageRowCandidates; }

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(ResolveImageRowCandidates);

ImageRowCandidatesKernel image_row_candidates_kernel() { return HWY_DYNAMIC_DISPATCH(ResolveImageRowCandidates)(); }

}  // namespace nss
#endif
