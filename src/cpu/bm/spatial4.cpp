#include "nss/cpu_api.hpp"
#include "cpu/hwy_config.hpp"
#include "cpu/bm/matcher.hpp"

#include <algorithm>
#include <cstdint>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/bm/spatial4.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

#if HWY_MAX_BYTES >= 64
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
static void SsdRow4Candidates16(const float* self, const float* candidates, int stride, int count,
                                float* distances) {
    const hn::FixedTag<float, 16> d16;
    for (int x = 0; x < count; x += 16) {
        const std::size_t lanes = static_cast<std::size_t>(std::min(16, count - x));
        auto acc0 = hn::Zero(d16);
        auto acc1 = hn::Zero(d16);
        auto acc2 = hn::Zero(d16);
        auto acc3 = hn::Zero(d16);
        for (int row = 0; row < 4; ++row) {
            const float* candidate_row = candidates + row * stride + x;
            const float* reference_row = self + row * stride;
            const auto d0 = hn::Sub(hn::Set(d16, reference_row[0]), hn::LoadN(d16, candidate_row, lanes));
            const auto d1 = hn::Sub(hn::Set(d16, reference_row[1]), hn::LoadN(d16, candidate_row + 1, lanes));
            const auto d2 = hn::Sub(hn::Set(d16, reference_row[2]), hn::LoadN(d16, candidate_row + 2, lanes));
            const auto d3 = hn::Sub(hn::Set(d16, reference_row[3]), hn::LoadN(d16, candidate_row + 3, lanes));
            acc0 = hn::MulAdd(d0, d0, acc0);
            acc1 = hn::MulAdd(d1, d1, acc1);
            acc2 = hn::MulAdd(d2, d2, acc2);
            acc3 = hn::MulAdd(d3, d3, acc3);
        }
        // Highway's 4-lane ReduceSum tree is (lane0 + lane3) +
        // (lane1 + lane2). Reproduce it across candidate lanes so the
        // matcher keeps the existing distance and tie semantics.
        const auto sums = hn::Add(hn::Add(acc0, acc3), hn::Add(acc1, acc2));
        hn::StoreN(sums, d16, distances + x, lanes);
    }
}
#endif

int SpatialMatch4Fast(const float* ref, int stride, int width, int height, int cx, int cy, int bm_range, int group,
                      Match* out) {
#if HWY_MAX_BYTES >= 16
    const hn::FixedTag<float, 4> df;
    const int wanted = std::min(group, kBmMaxGroup);
    const int top = std::max(cy - bm_range, 0);
    const int bottom = std::min(cy + bm_range, height - 4);
    const int left = std::max(cx - bm_range, 0);
    const int right = std::min(cx + bm_range, width - 4);
    const float* self = ref + cy * stride + cx;
    out[0] = Match{cx, cy, 0, 0.f, 0};
    if (wanted <= 1) {
        return 1;
    }

#if HWY_MAX_BYTES < 64
    hn::Vec<decltype(df)> refb[4];
    for (int i = 0; i < 4; ++i) {
        refb[i] = hn::LoadU(df, self + i * stride);
    }
#endif

    auto fallback = [&]() {
        return detail::collect_spatial(ref, stride, width, height, cx, cy, 4, bm_range, group, out,
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

    for (int y = top; y <= bottom && !nonfinite; ++y) {
        const float* row = ref + y * stride;
#if HWY_MAX_BYTES >= 64
        HWY_ALIGN float distances[2 * kBmMaxRange + 1];
        const int candidate_count = right - left + 1;
        SsdRow4Candidates16(self, row + left, stride, candidate_count, distances);
        for (int i = 0; i < candidate_count && !nonfinite; ++i) {
            consider(left + i, y, distances[i]);
        }
#else
        int x = left;
        for (; x + 1 <= right && !nonfinite; x += 2) {
            auto acc0 = hn::Zero(df);
            auto acc1 = hn::Zero(df);
            for (int i = 0; i < 4; ++i) {
                const auto d0 = hn::Sub(refb[i], hn::LoadU(df, row + x + i * stride));
                const auto d1 = hn::Sub(refb[i], hn::LoadU(df, row + x + 1 + i * stride));
                acc0 = hn::MulAdd(d0, d0, acc0);
                acc1 = hn::MulAdd(d1, d1, acc1);
            }
            consider(x, y, hn::ReduceSum(df, acc0));
            consider(x + 1, y, hn::ReduceSum(df, acc1));
        }
        if (x <= right && !nonfinite) {
            auto acc = hn::Zero(df);
            for (int i = 0; i < 4; ++i) {
                const auto d = hn::Sub(refb[i], hn::LoadU(df, row + x + i * stride));
                acc = hn::MulAdd(d, d, acc);
            }
            consider(x, y, hn::ReduceSum(df, acc));
        }
#endif
    }
    if (nonfinite) {
        return fallback();
    }
    return 1 + topk.finish();
#else
    return detail::collect_spatial(ref, stride, width, height, cx, cy, 4, bm_range, group, out,
                                   [](const float* a, const float* b, int st, int bs) {
                                       return nss::ssd_block(a, st, b, st, bs);
                                   });
#endif
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss::detail {
HWY_EXPORT(SpatialMatch4Fast);

int spatial_match4_fast(const float* ref, int stride, int width, int height, int cx, int cy, int bm_range, int group,
                        Match* out) {
    return HWY_DYNAMIC_DISPATCH(SpatialMatch4Fast)(ref, stride, width, height, cx, cy, bm_range, group, out);
}

}  // namespace nss::detail
#endif
