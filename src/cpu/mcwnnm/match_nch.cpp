#include "nss/cpu_api.hpp"
#include "nss/cpu_mcwnnm.hpp"
#include "cpu/hwy_config.hpp"
#include "cpu/bm/matcher.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/mcwnnm/match_nch.cpp"
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

static float DistNch(const float* const* refs, const int* strides, int nch, int cx, int cy, int x, int y, int block) {
    const float* a[3];
    const float* b[3];
    int sa[3];
    int sb[3];
    const int n = std::min(nch, 3);
    for (int c = 0; c < n; ++c) {
        if (!refs[c]) {
            a[c] = nullptr;
            b[c] = nullptr;
            sa[c] = 0;
            sb[c] = 0;
            continue;
        }
        a[c] = refs[c] + cy * strides[c] + cx;
        b[c] = refs[c] + y * strides[c] + x;
        sa[c] = strides[c];
        sb[c] = strides[c];
    }
    return ssd_nch(a, sa, b, sb, n, block);
}

int SpatialMatchNch(const float* const* refs, const int* strides, int nch, int width, int height, int bx, int by,
                    int block, int bm_range, int group, Match* out) {
    if (!refs || !strides || !out || nch < 1 || nch > 3 || width < 1 || height < 1 || block < 1 || group < 1 ||
        group > kBmMaxGroup || bm_range < 0) {
        return 0;
    }
    for (int c = 0; c < nch; ++c) {
        if (!refs[c] || strides[c] < width) {
            return 0;
        }
    }

#if HWY_MAX_BYTES >= 32
    if (nch == 3 && block == 8) {
        const hn::FixedTag<float, 8> d;
        const int max_x = width - 8;
        const int max_y = height - 8;
        if (max_x < 0 || max_y < 0) {
            return 0;
        }
        const int cx = std::clamp(bx, 0, max_x);
        const int cy = std::clamp(by, 0, max_y);
        hn::Vec<decltype(d)> query[3][8];
        for (int c = 0; c < 3; ++c) {
            const float* self = refs[c] + cy * strides[c] + cx;
            for (int y = 0; y < 8; ++y) {
                query[c][y] = hn::LoadU(d, self + y * strides[c]);
            }
        }
        return detail::collect_spatial_coords(
            width, height, bx, by, block, bm_range, group, out, [&](int /*qcx*/, int /*qcy*/, int x, int y) {
                auto acc0 = hn::Zero(d);
                auto acc1 = hn::Zero(d);
                auto acc2 = hn::Zero(d);
                for (int row = 0; row < 8; ++row) {
                    const auto e0 = hn::Sub(query[0][row], hn::LoadU(d, refs[0] + (y + row) * strides[0] + x));
                    const auto e1 = hn::Sub(query[1][row], hn::LoadU(d, refs[1] + (y + row) * strides[1] + x));
                    const auto e2 = hn::Sub(query[2][row], hn::LoadU(d, refs[2] + (y + row) * strides[2] + x));
                    acc0 = hn::MulAdd(e0, e0, acc0);
                    acc1 = hn::MulAdd(e1, e1, acc1);
                    acc2 = hn::MulAdd(e2, e2, acc2);
                }
                return HSum8(hn::Add(acc0, hn::Add(acc1, acc2)));
            });
    }
#endif

    return detail::collect_spatial_coords(width, height, bx, by, block, bm_range, group, out,
                                          [&](int cx, int cy, int x, int y) {
                                              return DistNch(refs, strides, nch, cx, cy, x, y, block);
                                          });
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(SpatialMatchNch);

int spatial_match_nch(const float* const* refs, const int* strides, int nch, int width, int height, int bx, int by,
                      int block, int bm_range, int group, Match* out) {
    return HWY_DYNAMIC_DISPATCH(SpatialMatchNch)(refs, strides, nch, width, height, bx, by, block, bm_range, group,
                                                 out);
}

int predictive_match_nch(const float* const* refs, const int* strides, int nch, int ntemp, int width, int height,
                         int bx, int by, int t0, const SearchConfig& cfg, Match* out) {
    if (!refs || !strides || !out || nch < 1 || nch > 3 || ntemp < 1 || t0 < 0 || t0 >= ntemp || width < 1 ||
        height < 1 || cfg.block < 1 || cfg.group < 1 || cfg.group > kBmMaxGroup || cfg.step < 1 || cfg.bm_range < 0 ||
        cfg.ps_num < 1 || cfg.ps_range < 0 || cfg.radius < 0 || cfg.radius > kBmMaxRadius) {
        return 0;
    }
    for (int c = 0; c < nch; ++c) {
        if (strides[c] < width) {
            return 0;
        }
        for (int t = 0; t < ntemp; ++t) {
            if (!refs[c * ntemp + t]) {
                return 0;
            }
        }
    }
    const float* cur[3];
    int cur_st[3];
    const int nch_use = std::min(nch, 3);
    for (int c = 0; c < nch_use; ++c) {
        cur[c] = refs[c * ntemp + t0];
        cur_st[c] = strides[c];
    }
    int n = spatial_match_nch(cur, cur_st, nch_use, width, height, bx, by, cfg.block, cfg.bm_range, cfg.group, out);
    std::uint32_t next_ordinal = 0;
    detail::assign_temporal_order(out, n, t0, t0, next_ordinal);
    if (ntemp <= 1 || cfg.radius <= 0 || n <= 0) {
        return n;
    }
    detail::StableTopK topk(out, cfg.group);
    topk.adopt(n);
    int px = out[0].x;
    int py = out[0].y;
    for (int dt = 1; dt <= cfg.radius; ++dt) {
        for (int sign = -1; sign <= 1; sign += 2) {
            const int t = t0 + sign * dt;
            if (t < 0 || t >= ntemp) {
                continue;
            }
            const float* fr[3];
            int fr_st[3];
            for (int c = 0; c < nch_use; ++c) {
                fr[c] = refs[c * ntemp + t];
                fr_st[c] = strides[c];
            }
            Match local[kBmMaxGroup];
            const int got = spatial_match_nch(fr, fr_st, nch_use, width, height, px, py, cfg.block, cfg.ps_range,
                                              std::min(cfg.ps_num + 1, cfg.group), local);
            for (int i = 0; i < got; ++i) {
                local[i].t = t;
                detail::assign_temporal_order(local + i, 1, t, t0, next_ordinal);
                topk.add(local[i]);
            }
            if (got > 0) {
                px = local[0].x;
                py = local[0].y;
            }
        }
    }
    return topk.finish();
}

}  // namespace nss
#endif
