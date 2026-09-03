#include "nss/cpu_api.hpp"
#include "nss/cpu_mcwnnm.hpp"
#include "cpu/bm/matcher.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace nss {
namespace {

float DistNch(const float* const* refs, const int* strides, int nch, int cx, int cy, int x, int y, int block) {
    float d = 0.f;
    for (int c = 0; c < nch; ++c) {
        if (!refs[c]) {
            continue;
        }
        d += ssd_block(refs[c] + cy * strides[c] + cx, strides[c], refs[c] + y * strides[c] + x, strides[c], block);
    }
    return d;
}

}  // namespace

int spatial_match_nch(const float* const* refs, const int* strides, int nch, int width, int height, int bx, int by,
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
    return detail::collect_spatial_coords(width, height, bx, by, block, bm_range, group, out,
                                          [&](int cx, int cy, int x, int y) {
                                              return DistNch(refs, strides, nch, cx, cy, x, y, block);
                                          });
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
