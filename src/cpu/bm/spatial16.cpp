#include "nss/cpu_api.hpp"
#include "cpu/hwy_config.hpp"
#include "cpu/bm/matcher.hpp"
#ifdef NSS_BM_KERNEL_LAB
#include "cpu/bm/kernel_lab.hpp"
#endif

#include <algorithm>
#include <cstdint>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/bm/spatial16.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;
#ifdef NSS_BM_KERNEL_LAB
#include "cpu/bm/ssd_row_lab-inl.hpp"
#endif

#if HWY_MAX_BYTES >= 64
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
static void SsdRow16(const float* self, const float* candidates, int stride, int count, float* distances) {
    const hn::FixedTag<float, 16> df;
    hn::Vec<decltype(df)> refb[16];
    for (int row = 0; row < 16; ++row) {
        refb[row] = hn::LoadU(df, self + row * stride);
    }
    int x = 0;
    for (; x + 1 < count; x += 2) {
        auto acc0 = hn::Zero(df);
        auto acc1 = hn::Zero(df);
        for (int row = 0; row < 16; ++row) {
            const auto d0 = hn::Sub(refb[row], hn::LoadU(df, candidates + x + row * stride));
            const auto d1 = hn::Sub(refb[row], hn::LoadU(df, candidates + x + 1 + row * stride));
            acc0 = hn::MulAdd(d0, d0, acc0);
            acc1 = hn::MulAdd(d1, d1, acc1);
        }
        distances[x] = hn::ReduceSum(df, acc0);
        distances[x + 1] = hn::ReduceSum(df, acc1);
    }
    if (x < count) {
        auto acc = hn::Zero(df);
        for (int row = 0; row < 16; ++row) {
            const auto diff = hn::Sub(refb[row], hn::LoadU(df, candidates + x + row * stride));
            acc = hn::MulAdd(diff, diff, acc);
        }
        distances[x] = hn::ReduceSum(df, acc);
    }
}
#endif

int SpatialMatch16Fast(const float* ref, int stride, int width, int height, int cx, int cy, int bm_range, int group,
                       Match* out) {
#if HWY_MAX_BYTES >= 64
    const int wanted = std::min(group, kBmMaxGroup);
    const int top = std::max(cy - bm_range, 0);
    const int bottom = std::min(cy + bm_range, height - 16);
    const int left = std::max(cx - bm_range, 0);
    const int right = std::min(cx + bm_range, width - 16);
    const float* self = ref + cy * stride + cx;
    out[0] = Match{cx, cy, 0, 0.f, 0};
    if (wanted <= 1) {
        return 1;
    }

    auto fallback = [&]() {
        return detail::collect_spatial(ref, stride, width, height, cx, cy, 16, bm_range, group, out,
                                       [](const float* a, const float* b, int st, int bs) {
                                           return nss::ssd_block(a, st, b, st, bs);
                                       });
    };

    detail::SortedTopK topk(out + 1, wanted - 1);
    bool nonfinite = false;
    const int span = right - left + 1;
    auto consider = [&](int x, int y, float dist) {
        if (nonfinite || (x == cx && y == cy)) {
            return;
        }
        if (!detail::finite_distance(dist)) {
            nonfinite = true;
            return;
        }
        const std::uint32_t ordinal = static_cast<std::uint32_t>((y - top) * span + (x - left) + 1);
        topk.add(Match{x, y, 0, dist, ordinal});
    };

    HWY_ALIGN float distances[2 * kBmMaxRange + 1];
    const int candidate_count = right - left + 1;
    for (int y = top; y <= bottom && !nonfinite; ++y) {
        const float* row = ref + y * stride;
        SsdRow16(self, row + left, stride, candidate_count, distances);
        for (int i = 0; i < candidate_count && !nonfinite; ++i) {
            consider(left + i, y, distances[i]);
        }
    }
    if (nonfinite) {
        return fallback();
    }
    return 1 + topk.finish();
#else
    return detail::collect_spatial(ref, stride, width, height, cx, cy, 16, bm_range, group, out,
                                   [](const float* a, const float* b, int st, int bs) {
                                       return nss::ssd_block(a, st, b, st, bs);
                                   });
#endif
}

#ifdef NSS_BM_KERNEL_LAB
detail::SsdRowKernel Ssd16LabKernel(int variant) {
#if HWY_MAX_BYTES >= 64
    switch (variant) {
        case 0: return &SsdRow16;
        case 1: return &SsdRowSplit8<16>;
        case 2: return &SsdRowStream<16>;
    }
#endif
    return nullptr;
}
#endif
}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss::detail {
HWY_EXPORT(SpatialMatch16Fast);
#ifdef NSS_BM_KERNEL_LAB
HWY_EXPORT(Ssd16LabKernel);
SsdRowKernel ssd16_lab_kernel(int variant) {
    return HWY_DYNAMIC_DISPATCH(Ssd16LabKernel)(variant);
}
#endif

int spatial_match16_fast(const float* ref, int stride, int width, int height, int cx, int cy, int bm_range, int group,
                         Match* out) {
    return HWY_DYNAMIC_DISPATCH(SpatialMatch16Fast)(ref, stride, width, height, cx, cy, bm_range, group, out);
}

}  // namespace nss::detail
#endif
