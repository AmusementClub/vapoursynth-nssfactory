#include "nss/cpu_api.hpp"
#include "nss/cpu_batch.hpp"
#include "nss/cpu_nlh.hpp"
#include "cpu/hwy_config.hpp"
#include "cpu/bm/matcher.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/nlh/match16.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

#if HWY_MAX_BYTES >= 32
static float HSum8(hn::Vec<hn::FixedTag<float, 8>> v) {
    const hn::FixedTag<float, 4> d4;
    return hn::ReduceSum(d4, hn::Add(hn::LowerHalf(d4, v), hn::UpperHalf(d4, v)));
}
#endif

static float Ssd8(const float* a, int sa, const float* b, int sb) {
#if HWY_MAX_BYTES >= 32
    const hn::FixedTag<float, 8> d;
    auto acc = hn::Zero(d);
    for (int y = 0; y < 8; ++y) {
        const auto diff = hn::Sub(hn::LoadU(d, a + y * sa), hn::LoadU(d, b + y * sb));
        acc = hn::MulAdd(diff, diff, acc);
    }
    return HSum8(acc);
#else
    float acc = 0.f;
    for (int y = 0; y < 8; ++y) {
        const float* pa = a + y * sa;
        const float* pb = b + y * sb;
        for (int x = 0; x < 8; ++x) {
            const float value = pa[x] - pb[x];
            acc += value * value;
        }
    }
    return acc;
#endif
}

static float SsdBlock(const float* a, int stride, const float* b, int block) {
    if (block == 8) {
        return Ssd8(a, stride, b, stride);
    }
    const hn::ScalableTag<float> d;
    const int lanes = static_cast<int>(hn::Lanes(d));
    auto acc = hn::Zero(d);
    for (int y = 0; y < block; ++y) {
        const float* pa = a + y * stride;
        const float* pb = b + y * stride;
        int x = 0;
        for (; x + lanes <= block; x += lanes) {
            const auto diff = hn::Sub(hn::LoadU(d, pa + x), hn::LoadU(d, pb + x));
            acc = hn::MulAdd(diff, diff, acc);
        }
        const int remaining = block - x;
        if (remaining > 0) {
            const auto diff = hn::Sub(hn::LoadN(d, pa + x, static_cast<std::size_t>(remaining)),
                                      hn::LoadN(d, pb + x, static_cast<std::size_t>(remaining)));
            acc = hn::MulAdd(diff, diff, acc);
        }
    }
    return hn::ReduceSum(d, acc);
}

int NlhSpatialMatch16(const float* ref, int stride, int width, int height, int bx, int by, int block, int bm_range,
                      Match* out) {
    const int max_x = width - block;
    const int max_y = height - block;
    if (!ref || !out || width < 1 || height < 1 || block < 1 || stride < width || max_x < 0 || max_y < 0) {
        return 0;
    }

    const int cx = std::clamp(bx, 0, max_x);
    const int cy = std::clamp(by, 0, max_y);
    const int range = std::max(bm_range, 0);
    const int top = std::max(cy - range, 0);
    const int bottom = std::min(cy + range, max_y);
    const int left = std::max(cx - range, 0);
    const int right = std::min(cx + range, max_x);
    const float* self = ref + cy * stride + cx;

    out[0] = Match{cx, cy, 0, 0.f, 0};
    Match* const sorted = out + 1;
    constexpr int capacity = 15;
    int size = 0;
    std::uint32_t ordinal = 1;
    for (int y = top; y <= bottom; ++y) {
        for (int x = left; x <= right; ++x, ++ordinal) {
            if (x == cx && y == cy) {
                continue;
            }
            const Match candidate{x, y, 0, SsdBlock(self, stride, ref + y * stride + x, block), ordinal};
            if (size == capacity && !detail::match_less(candidate, sorted[size - 1])) {
                continue;
            }
            int pos = std::min(size, capacity - 1);
            while (pos > 0 && detail::match_less(candidate, sorted[pos - 1])) {
                if (pos < capacity) {
                    sorted[pos] = sorted[pos - 1];
                }
                --pos;
            }
            sorted[pos] = candidate;
            if (size < capacity) {
                ++size;
            }
        }
    }
    return size + 1;
}

int NlhSpatialMatch16Batch(const float* ref, int stride, int width, int height, const MatchBatchItem* items, int count,
                           Match* matches, int match_stride, int* counts) {
    if (count == 0) {
        return 0;
    }
    if (!ref || !items || !matches || !counts || count < 0 || match_stride < 1) {
        return -1;
    }
    int first_error = 0;
    for (int i = 0; i < count; ++i) {
        const auto& item = items[i];
        counts[i] = NlhSpatialMatch16(ref, stride, width, height, item.bx, item.by, item.block, item.bm_range,
                                      matches + static_cast<std::size_t>(i) * match_stride);
        if (counts[i] <= 0 && first_error == 0) {
            first_error = i + 1;
        }
    }
    return first_error;
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(NlhSpatialMatch16);
HWY_EXPORT(NlhSpatialMatch16Batch);

int nlh_spatial_match16(const float* ref, int stride, int width, int height, int bx, int by, int block, int bm_range,
                        Match* out) {
    return HWY_DYNAMIC_DISPATCH(NlhSpatialMatch16)(ref, stride, width, height, bx, by, block, bm_range, out);
}

int nlh_spatial_match_batch(const float* ref, int stride, int width, int height, const MatchBatchItem* items,
                            int count, Match* matches, int match_stride, int* counts) {
    if (count <= 0 || !items) {
        return spatial_match_batch(ref, stride, width, height, items, count, matches, match_stride, counts);
    }
    const int block = items[0].block;
    const int bm_range = items[0].bm_range;
    for (int i = 0; i < count; ++i) {
        if (items[i].group != 16 || items[i].block != block || items[i].bm_range != bm_range) {
            return spatial_match_batch(ref, stride, width, height, items, count, matches, match_stride, counts);
        }
    }
    return HWY_DYNAMIC_DISPATCH(NlhSpatialMatch16Batch)(ref, stride, width, height, items, count, matches,
                                                        match_stride, counts);
}

}  // namespace nss
#endif
