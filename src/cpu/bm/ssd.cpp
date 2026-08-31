#include "nss/cpu_api.hpp"
#include "cpu/hwy_config.hpp"

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
    if (block == 8) {
        return Ssd8(a, sa, b, sb);
    }
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
    auto errors = hn::Set(df, std::numeric_limits<float>::infinity());
    auto vx = hn::Set(di, cx);
    auto vy = hn::Set(di, cy);
    const auto iota = hn::Iota(di, 0);
    const auto one = hn::Set(di, 1);
    for (int y = top; y <= bottom; ++y) {
        const float* row = ref + y * stride;
        for (int x = left; x <= right; ++x) {
            auto acc0 = hn::Zero(df);
            auto acc1 = hn::Zero(df);
            for (int i = 0; i < 8; i += 2) {
                const auto d0 = hn::Sub(refb[i], hn::LoadU(df, row + x + i * stride));
                const auto d1 = hn::Sub(refb[i + 1], hn::LoadU(df, row + x + (i + 1) * stride));
                acc0 = hn::MulAdd(d0, d0, acc0);
                acc1 = hn::MulAdd(d1, d1, acc1);
            }
            const float dist = HSum8(hn::Add(acc0, acc1));
            const auto vdist = hn::Set(df, dist);
            const int count = static_cast<int>(hn::CountTrue(df, hn::Lt(vdist, errors)));
            if (count == 0) {
                continue;
            }
            const int pos = 8 - count;
            const auto posi = hn::Set(di, pos);
            const auto src = hn::IfThenElse(hn::Gt(iota, posi), hn::Sub(iota, one), iota);
            const auto idxf = hn::IndicesFromVec(df, hn::BitCast(du, src));
            const auto idxi = hn::IndicesFromVec(di, hn::BitCast(du, src));
            const auto at = hn::RebindMask(df, hn::Eq(iota, posi));
            errors = hn::IfThenElse(at, vdist, hn::TableLookupLanes(errors, idxf));
            vx = hn::IfThenElse(hn::Eq(iota, posi), hn::Set(di, x), hn::TableLookupLanes(vx, idxi));
            vy = hn::IfThenElse(hn::Eq(iota, posi), hn::Set(di, y), hn::TableLookupLanes(vy, idxi));
        }
    }
    HWY_ALIGN float dists[8];
    HWY_ALIGN std::int32_t xs[8];
    HWY_ALIGN std::int32_t ys[8];
    hn::Store(errors, df, dists);
    hn::Store(vx, di, xs);
    hn::Store(vy, di, ys);
    int n = 0;
    for (int i = 0; i < 8; ++i) {
        if (!std::isfinite(dists[i])) {
            break;
        }
        out[n].x = xs[i];
        out[n].y = ys[i];
        out[n].t = 0;
        out[n].dist = dists[i];
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

static float Ssd8FromRef(const hn::Vec<hn::FixedTag<float, 8>> refb[8], const float* cand, int stride, float abort_at) {
    const hn::FixedTag<float, 8> df;
    auto acc0 = hn::Zero(df);
    auto acc1 = hn::Zero(df);
    for (int i = 0; i < 4; i += 2) {
        const auto d0 = hn::Sub(refb[i], hn::LoadU(df, cand + i * stride));
        const auto d1 = hn::Sub(refb[i + 1], hn::LoadU(df, cand + (i + 1) * stride));
        acc0 = hn::MulAdd(d0, d0, acc0);
        acc1 = hn::MulAdd(d1, d1, acc1);
    }
    if (abort_at >= 0.f) {
        const float p = HSum8(hn::Add(acc0, acc1));
        if (p >= abort_at) {
            return p;
        }
    }
    for (int i = 4; i < 8; i += 2) {
        const auto d0 = hn::Sub(refb[i], hn::LoadU(df, cand + i * stride));
        const auto d1 = hn::Sub(refb[i + 1], hn::LoadU(df, cand + (i + 1) * stride));
        acc0 = hn::MulAdd(d0, d0, acc0);
        acc1 = hn::MulAdd(d1, d1, acc1);
    }
    return HSum8(hn::Add(acc0, acc1));
}
#endif

int SpatialMatch(const float* ref, int stride, int width, int height, int bx, int by, int block, int bm_range,
                 int group, Match* out) {
    const int max_x = width - block;
    const int max_y = height - block;
    if (max_x < 0 || max_y < 0 || group < 1) {
        return 0;
    }
    const int cx = clampi(bx, 0, max_x);
    const int cy = clampi(by, 0, max_y);
#if HWY_MAX_BYTES >= 32
    if (block == 8 && group == 8) {
        return SpatialMatch8(ref, stride, width, height, cx, cy, bm_range, out);
    }
#endif
    const float* self = ref + cy * stride + cx;
    out[0].x = cx;
    out[0].y = cy;
    out[0].t = 0;
    out[0].dist = 0.f;
    if (group <= 1) {
        return 1;
    }

    const int top = std::max(cy - bm_range, 0);
    const int bottom = std::min(cy + bm_range, max_y);
    const int left = std::max(cx - bm_range, 0);
    const int right = std::min(cx + bm_range, max_x);

    int n = 1;
    int worst = 1;
    auto consider = [&](int x, int y, float dist) {
        if (x == cx && y == cy) {
            return;
        }
        if (n < group) {
            out[n].x = x;
            out[n].y = y;
            out[n].t = 0;
            out[n].dist = dist;
            if (n == 1 || dist > out[worst].dist) {
                worst = n;
            }
            ++n;
        } else if (__builtin_expect(dist < out[worst].dist, 0)) {
            out[worst].x = x;
            out[worst].y = y;
            out[worst].t = 0;
            out[worst].dist = dist;
            worst = 1;
            for (int i = 2; i < n; ++i) {
                if (out[i].dist > out[worst].dist) {
                    worst = i;
                }
            }
        }
    };
#if HWY_MAX_BYTES >= 32
    if (block == 8) {
        const hn::FixedTag<float, 8> df;
        hn::Vec<hn::FixedTag<float, 8>> refb[8];
        for (int i = 0; i < 8; ++i) {
            refb[i] = hn::LoadU(df, self + i * stride);
        }
        for (int y = top; y <= bottom; ++y) {
            const float* row = ref + y * stride;
            for (int x = left; x <= right; ++x) {
                const float abort_at = (n >= group) ? out[worst].dist : -1.f;
                consider(x, y, Ssd8FromRef(refb, row + x, stride, abort_at));
            }
        }
        return n;
    }
#endif
    for (int y = top; y <= bottom; ++y) {
        for (int x = left; x <= right; ++x) {
            consider(x, y, SsdBlock(self, stride, ref + y * stride + x, stride, block));
        }
    }
    return n;
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(SsdBlock);
HWY_EXPORT(SpatialMatch);

float ssd_block(const float* a, int sa, const float* b, int sb, int block) {
    return HWY_DYNAMIC_DISPATCH(SsdBlock)(a, sa, b, sb, block);
}

int spatial_match(const float* ref, int stride, int width, int height, int bx, int by, int block, int bm_range,
                  int group, Match* out) {
    return HWY_DYNAMIC_DISPATCH(SpatialMatch)(ref, stride, width, height, bx, by, block, bm_range, group, out);
}

}  // namespace nss
#endif
