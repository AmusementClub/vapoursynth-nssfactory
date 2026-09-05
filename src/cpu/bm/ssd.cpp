#include "nss/cpu_api.hpp"
#include "cpu/hwy_config.hpp"
#include "cpu/bm/matcher.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/bm/ssd.cpp"
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

#if HWY_MAX_BYTES >= 16
static float Ssd4(const float* a, int sa, const float* b, int sb) {
    const hn::FixedTag<float, 4> d;
    auto acc = hn::Zero(d);
    for (int y = 0; y < 4; ++y) {
        const auto diff = hn::Sub(hn::LoadU(d, a + y * sa), hn::LoadU(d, b + y * sb));
        acc = hn::MulAdd(diff, diff, acc);
    }
    return hn::ReduceSum(d, acc);
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
            const float t = pa[x] - pb[x];
            acc += t * t;
        }
    }
    return acc;
#endif
}

float SsdBlock(const float* a, int sa, const float* b, int sb, int block) {
    if (!a || !b || block < 1 || sa < block || sb < block) {
        // Invalid views are rejected by the public matcher; use a finite
        // sentinel here so this fast-math TU does not rely on an infinity
        // constant whose classification can be folded away.
        return std::numeric_limits<float>::max();
    }
    if (block == 8) {
        return Ssd8(a, sa, b, sb);
    }
#if HWY_MAX_BYTES >= 16
    if (block == 4) {
        return Ssd4(a, sa, b, sb);
    }
#endif
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    auto acc = hn::Zero(d);
    for (int y = 0; y < block; ++y) {
        const float* pa = a + y * sa;
        const float* pb = b + y * sb;
        int x = 0;
        for (; x + N <= block; x += N) {
            const auto diff = hn::Sub(hn::LoadU(d, pa + x), hn::LoadU(d, pb + x));
            acc = hn::MulAdd(diff, diff, acc);
        }
        const int rem = block - x;
        if (rem > 0) {
            const auto diff =
                hn::Sub(hn::LoadN(d, pa + x, static_cast<size_t>(rem)), hn::LoadN(d, pb + x, static_cast<size_t>(rem)));
            acc = hn::MulAdd(diff, diff, acc);
        }
    }
    return hn::ReduceSum(d, acc);
}

float SsdNch(const float* const* a, const int* sa, const float* const* b, const int* sb, int nch, int block) {
    if (!a || !sa || !b || !sb || nch < 1 || block < 1) {
        return std::numeric_limits<float>::max();
    }
#if HWY_MAX_BYTES >= 32
    if (nch == 3 && block == 8 && a[0] && a[1] && a[2] && b[0] && b[1] && b[2] && sa[0] >= 8 && sa[1] >= 8 &&
        sa[2] >= 8 && sb[0] >= 8 && sb[1] >= 8 && sb[2] >= 8) {
        const hn::FixedTag<float, 8> d;
        auto acc0 = hn::Zero(d);
        auto acc1 = hn::Zero(d);
        auto acc2 = hn::Zero(d);
        for (int y = 0; y < 8; ++y) {
            const auto d0 = hn::Sub(hn::LoadU(d, a[0] + y * sa[0]), hn::LoadU(d, b[0] + y * sb[0]));
            const auto d1 = hn::Sub(hn::LoadU(d, a[1] + y * sa[1]), hn::LoadU(d, b[1] + y * sb[1]));
            const auto d2 = hn::Sub(hn::LoadU(d, a[2] + y * sa[2]), hn::LoadU(d, b[2] + y * sb[2]));
            acc0 = hn::MulAdd(d0, d0, acc0);
            acc1 = hn::MulAdd(d1, d1, acc1);
            acc2 = hn::MulAdd(d2, d2, acc2);
        }
        return HSum8(hn::Add(acc0, hn::Add(acc1, acc2)));
    }
#endif
    float acc = 0.f;
    for (int c = 0; c < nch; ++c) {
        if (!a[c] || !b[c]) {
            continue;
        }
        acc += SsdBlock(a[c], sa[c], b[c], sb[c], block);
    }
    return acc;
}

static int clampi(int x, int lo, int hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

#if HWY_MAX_BYTES >= 32
// Sorted-ascending top-8 insert, same idea as bm3dcpu; only for block=8, group=8.
static int SpatialMatch8(const float* ref, int stride, int width, int height, int cx, int cy, int bm_range,
                         Match* out) {
    const hn::FixedTag<float, 8> df;
    const hn::FixedTag<std::int32_t, 8> di;
    const hn::RebindToUnsigned<decltype(di)> du;
    const int max_x = width - 8;
    const int max_y = height - 8;
    const int top = std::max(cy - bm_range, 0);
    const int bottom = std::min(cy + bm_range, max_y);
    const int left = std::max(cx - bm_range, 0);
    const int right = std::min(cx + bm_range, max_x);
    const float* self = ref + cy * stride + cx;
    hn::Vec<decltype(df)> refb[8];
    for (int i = 0; i < 8; ++i) {
        refb[i] = hn::LoadU(df, self + i * stride);
    }
    // Keep the self match in lane zero.  The explicit fill count below is
    // needed because this TU uses fast-math, where isfinite(infinity) is not
    // a reliable way to identify an unused lane.
    HWY_ALIGN float initial_errors[8] = {0.f,
                                         std::numeric_limits<float>::max(),
                                         std::numeric_limits<float>::max(),
                                         std::numeric_limits<float>::max(),
                                         std::numeric_limits<float>::max(),
                                         std::numeric_limits<float>::max(),
                                         std::numeric_limits<float>::max(),
                                         std::numeric_limits<float>::max()};
    auto errors = hn::Load(df, initial_errors);
    auto vx = hn::Set(di, cx);
    auto vy = hn::Set(di, cy);
    const auto iota = hn::Iota(di, 0);
    const auto one = hn::Set(di, 1);
    int filled = 1;
    float worst = std::numeric_limits<float>::max();
    auto insert = [&](int x, int y, hn::Vec<decltype(df)> vdist, int count) {
        const int pos = 8 - count;
        const auto posi = hn::Set(di, pos);
        const auto src = hn::IfThenElse(hn::Gt(iota, posi), hn::Sub(iota, one), iota);
        const auto idxf = hn::IndicesFromVec(df, hn::BitCast(du, src));
        const auto idxi = hn::IndicesFromVec(di, hn::BitCast(du, src));
        const auto at = hn::RebindMask(df, hn::Eq(iota, posi));
        errors = hn::IfThenElse(at, vdist, hn::TableLookupLanes(errors, idxf));
        vx = hn::IfThenElse(hn::Eq(iota, posi), hn::Set(di, x), hn::TableLookupLanes(vx, idxi));
        vy = hn::IfThenElse(hn::Eq(iota, posi), hn::Set(di, y), hn::TableLookupLanes(vy, idxi));
    };
    auto consider = [&](int x, int y, float dist) {
        if (x == cx && y == cy) {
            return;
        }
        if (filled < 8) {
            const auto vdist = hn::Set(df, dist);
            const int count = static_cast<int>(hn::CountTrue(df, hn::Lt(vdist, errors)));
            if (count != 0) {
                insert(x, y, vdist, count);
                ++filled;
            }
            if (filled == 8) {
                worst = hn::ExtractLane(errors, 7);
            }
        } else if (dist < worst) {
            const auto vdist = hn::Set(df, dist);
            const int count = static_cast<int>(hn::CountTrue(df, hn::Lt(vdist, errors)));
            insert(x, y, vdist, count);
            worst = hn::ExtractLane(errors, 7);
        }
    };
    auto fallback = [&]() {
        return detail::collect_spatial(ref, stride, width, height, cx, cy, 8, bm_range, 8, out,
                                       [](const float* a, const float* b, int st, int bs) {
                                           return SsdBlock(a, st, b, st, bs);
                                       });
    };
    for (int y = top; y <= bottom; ++y) {
        const float* row = ref + y * stride;
        int x = left;
        for (; x + 1 <= right; x += 2) {
            auto a00 = hn::Zero(df);
            auto a01 = hn::Zero(df);
            auto a02 = hn::Zero(df);
            auto a03 = hn::Zero(df);
            auto a10 = hn::Zero(df);
            auto a11 = hn::Zero(df);
            auto a12 = hn::Zero(df);
            auto a13 = hn::Zero(df);
            for (int i = 0; i < 8; i += 4) {
                const auto d00 = hn::Sub(refb[i], hn::LoadU(df, row + x + i * stride));
                const auto d01 = hn::Sub(refb[i + 1], hn::LoadU(df, row + x + (i + 1) * stride));
                const auto d02 = hn::Sub(refb[i + 2], hn::LoadU(df, row + x + (i + 2) * stride));
                const auto d03 = hn::Sub(refb[i + 3], hn::LoadU(df, row + x + (i + 3) * stride));
                const auto d10 = hn::Sub(refb[i], hn::LoadU(df, row + x + 1 + i * stride));
                const auto d11 = hn::Sub(refb[i + 1], hn::LoadU(df, row + x + 1 + (i + 1) * stride));
                const auto d12 = hn::Sub(refb[i + 2], hn::LoadU(df, row + x + 1 + (i + 2) * stride));
                const auto d13 = hn::Sub(refb[i + 3], hn::LoadU(df, row + x + 1 + (i + 3) * stride));
                a00 = hn::MulAdd(d00, d00, a00);
                a01 = hn::MulAdd(d01, d01, a01);
                a02 = hn::MulAdd(d02, d02, a02);
                a03 = hn::MulAdd(d03, d03, a03);
                a10 = hn::MulAdd(d10, d10, a10);
                a11 = hn::MulAdd(d11, d11, a11);
                a12 = hn::MulAdd(d12, d12, a12);
                a13 = hn::MulAdd(d13, d13, a13);
            }
            const float dist0 = HSum8(hn::Add(hn::Add(a00, a02), hn::Add(a01, a03)));
            const float dist1 = HSum8(hn::Add(hn::Add(a10, a12), hn::Add(a11, a13)));
            consider(x, y, dist0);
            consider(x + 1, y, dist1);
        }
        if (x <= right) {
            auto acc0 = hn::Zero(df);
            auto acc1 = hn::Zero(df);
            auto acc2 = hn::Zero(df);
            auto acc3 = hn::Zero(df);
            for (int i = 0; i < 8; i += 4) {
                const auto d0 = hn::Sub(refb[i], hn::LoadU(df, row + x + i * stride));
                const auto d1 = hn::Sub(refb[i + 1], hn::LoadU(df, row + x + (i + 1) * stride));
                const auto d2 = hn::Sub(refb[i + 2], hn::LoadU(df, row + x + (i + 2) * stride));
                const auto d3 = hn::Sub(refb[i + 3], hn::LoadU(df, row + x + (i + 3) * stride));
                acc0 = hn::MulAdd(d0, d0, acc0);
                acc1 = hn::MulAdd(d1, d1, acc1);
                acc2 = hn::MulAdd(d2, d2, acc2);
                acc3 = hn::MulAdd(d3, d3, acc3);
            }
            const float dist = HSum8(hn::Add(hn::Add(acc0, acc2), hn::Add(acc1, acc3)));
            consider(x, y, dist);
        }
    }
    std::uint32_t max_bits = hn::ReduceMax(du, hn::BitCast(du, errors));
#if defined(__GNUC__) || defined(__clang__)
    __asm__ volatile("" : "+r"(max_bits));
#else
    volatile std::uint32_t retained_max_bits = max_bits;
    max_bits = retained_max_bits;
#endif
    // Non-finite SSD results cannot pass the ordered lane comparison. The
    // fill count proves that enough candidates displaced the FLT_MAX
    // sentinels; the integer reduction keeps that conclusion valid under
    // -ffinite-math-only and also catches a retained NaN or infinity.
    const int candidate_count = (right - left + 1) * (bottom - top + 1) - 1;
    const int expected = 1 + std::min(7, std::max(candidate_count, 0));
    if (filled != expected || max_bits >= 0x7f800000u) {
        return fallback();
    }
    HWY_ALIGN float dists[8];
    HWY_ALIGN std::int32_t xs[8];
    HWY_ALIGN std::int32_t ys[8];
    hn::Store(errors, df, dists);
    hn::Store(vx, di, xs);
    hn::Store(vy, di, ys);
    int n = 0;
    for (int i = 0; i < filled; ++i) {
        out[n].x = xs[i];
        out[n].y = ys[i];
        out[n].t = 0;
        out[n].dist = dists[i];
        const int span = right - left + 1;
        out[n].ordinal = (xs[i] == cx && ys[i] == cy)
                             ? 0u
                             : static_cast<std::uint32_t>((ys[i] - top) * span + (xs[i] - left) + 1);
        ++n;
    }
    if (n == 0) {
        out[0].x = cx;
        out[0].y = cy;
        out[0].t = 0;
        out[0].dist = 0.f;
        return 1;
    }
    return n;
}

// Same hoisted 8x8 SSD as SpatialMatch8, but top-k length follows group.
// group==8 keeps the lane-sorted path above; this is only the K-generic sibling.
static int SpatialMatch8Ssd(const float* ref, int stride, int width, int height, int cx, int cy, int bm_range,
                            int group, Match* out) {
    const hn::FixedTag<float, 8> df;
    const int wanted = std::min(group, kBmMaxGroup);
    const int top = std::max(cy - bm_range, 0);
    const int bottom = std::min(cy + bm_range, height - 8);
    const int left = std::max(cx - bm_range, 0);
    const int right = std::min(cx + bm_range, width - 8);
    const float* self = ref + cy * stride + cx;
    out[0] = Match{cx, cy, 0, 0.f, 0};
    if (wanted <= 1) {
        return 1;
    }

    hn::Vec<decltype(df)> refb[8];
    for (int i = 0; i < 8; ++i) {
        refb[i] = hn::LoadU(df, self + i * stride);
    }

    auto fallback = [&]() {
        return detail::collect_spatial(ref, stride, width, height, cx, cy, 8, bm_range, group, out,
                                       [](const float* a, const float* b, int st, int bs) {
                                           return SsdBlock(a, st, b, st, bs);
                                       });
    };

    detail::StableTopK topk(out + 1, wanted - 1);
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
        int x = left;
        for (; x + 1 <= right && !nonfinite; x += 2) {
            // Same reduction tree as Ssd8 so group!=8 stays bit-exact vs collect_spatial.
            auto acc0 = hn::Zero(df);
            auto acc1 = hn::Zero(df);
            for (int i = 0; i < 8; ++i) {
                const auto d0 = hn::Sub(refb[i], hn::LoadU(df, row + x + i * stride));
                const auto d1 = hn::Sub(refb[i], hn::LoadU(df, row + x + 1 + i * stride));
                acc0 = hn::MulAdd(d0, d0, acc0);
                acc1 = hn::MulAdd(d1, d1, acc1);
            }
            consider(x, y, HSum8(acc0));
            consider(x + 1, y, HSum8(acc1));
        }
        if (x <= right && !nonfinite) {
            auto acc = hn::Zero(df);
            for (int i = 0; i < 8; ++i) {
                const auto d = hn::Sub(refb[i], hn::LoadU(df, row + x + i * stride));
                acc = hn::MulAdd(d, d, acc);
            }
            consider(x, y, HSum8(acc));
        }
    }
    if (nonfinite) {
        return fallback();
    }
    return 1 + topk.finish();
}

#endif

int SpatialMatch(const float* ref, int stride, int width, int height, int bx, int by, int block, int bm_range,
                 int group, Match* out) {
    const int max_x = width - block;
    const int max_y = height - block;
    if (!ref || !out || width < 1 || height < 1 || block < 1 || stride < width || max_x < 0 || max_y < 0 ||
        group < 1) {
        return 0;
    }
    const int cx = clampi(bx, 0, max_x);
    const int cy = clampi(by, 0, max_y);
#if HWY_MAX_BYTES >= 32
    if (block == 8 && group == 8) {
        return SpatialMatch8(ref, stride, width, height, cx, cy, std::max(bm_range, 0), out);
    }
    if (block == 8) {
        return SpatialMatch8Ssd(ref, stride, width, height, cx, cy, std::max(bm_range, 0), group, out);
    }
#endif
#if HWY_MAX_BYTES >= 16
    if (block == 4) {
        return detail::spatial_match4_fast(ref, stride, width, height, cx, cy, std::max(bm_range, 0), group, out);
    }
#endif
#ifndef NSS_DISABLE_BM3D_B12_FAST
    if (block == 12) {
        return detail::spatial_match12_fast(ref, stride, width, height, cx, cy, std::max(bm_range, 0), group, out);
    }
#endif
    if (block == 16) {
        return detail::spatial_match16_fast(ref, stride, width, height, cx, cy, std::max(bm_range, 0), group, out);
    }
    return detail::collect_spatial(ref, stride, width, height, cx, cy, block, bm_range, group, out,
                                   [](const float* a, const float* b, int st, int bs) {
                                       return SsdBlock(a, st, b, st, bs);
                                   });
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(SsdBlock);
HWY_EXPORT(SsdNch);
HWY_EXPORT(SpatialMatch);

float ssd_block(const float* a, int sa, const float* b, int sb, int block) {
    return HWY_DYNAMIC_DISPATCH(SsdBlock)(a, sa, b, sb, block);
}

float ssd_nch(const float* const* a, const int* sa, const float* const* b, const int* sb, int nch, int block) {
    return HWY_DYNAMIC_DISPATCH(SsdNch)(a, sa, b, sb, nch, block);
}

int spatial_match(const float* ref, int stride, int width, int height, int bx, int by, int block, int bm_range,
                  int group, Match* out) {
    return HWY_DYNAMIC_DISPATCH(SpatialMatch)(ref, stride, width, height, bx, by, block, bm_range, group, out);
}

}  // namespace nss
#endif
