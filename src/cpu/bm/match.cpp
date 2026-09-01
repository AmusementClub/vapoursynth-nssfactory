#include "nss/cpu_api.hpp"
#include "cpu/bm/matcher.hpp"

#include <algorithm>
#include <cstdint>

namespace nss {

int predictive_match(const float* const* refs, const int* strides, int ntemp, int width, int height, int bx, int by,
                     int t0, const SearchConfig& cfg, Match* out) {
    if (!refs || !strides || !out || ntemp < 1 || t0 < 0 || t0 >= ntemp || width < 1 || height < 1 || cfg.block < 1 ||
        cfg.group < 1 || cfg.group > kBmMaxGroup || cfg.step < 1 || cfg.bm_range < 0 || cfg.ps_num < 1 ||
        cfg.ps_range < 0 || cfg.radius < 0 || cfg.radius > kBmMaxRadius || !refs[t0] || strides[t0] < width) {
        return 0;
    }
    for (int t = 0; t < ntemp; ++t) {
        if (!refs[t] || strides[t] < width) {
            return 0;
        }
    }
    const int block = cfg.block;
    const int group = cfg.group;
    int n = spatial_match(refs[t0], strides[t0], width, height, bx, by, block, cfg.bm_range, group, out);
    std::uint32_t next_ordinal = 0;
    detail::assign_temporal_order(out, n, t0, t0, next_ordinal);
    if (ntemp <= 1 || cfg.radius <= 0 || n <= 0) {
        return n;
    }

    constexpr int kCap = kBmMaxGroup * (1 + 2 * kBmMaxRadius);
    Match all[kCap];
    int nall = n;
    if (nall > kCap) {
        nall = kCap;
    }
    for (int i = 0; i < nall; ++i) {
        all[i] = out[i];
    }

    int px = out[0].x;
    int py = out[0].y;
    for (int dt = 1; dt <= cfg.radius; ++dt) {
        for (int sign = -1; sign <= 1; sign += 2) {
            const int t = t0 + sign * dt;
            if (t < 0 || t >= ntemp) {
                continue;
            }
            Match local[kBmMaxGroup];
            const int got = spatial_match(refs[t], strides[t], width, height, px, py, block, cfg.ps_range,
                                          std::min(cfg.ps_num + 1, group), local);
            for (int i = 0; i < got; ++i) {
                local[i].t = t;
                // spatial_match has already assigned the original traversal
                // ordinal; only add the temporal frame component here.
                detail::assign_temporal_order(local + i, 1, t, t0, next_ordinal);
                if (nall < kCap) {
                    all[nall++] = local[i];
                }
            }
            if (got > 0) {
                px = local[0].x;
                py = local[0].y;
            }
        }
    }
    const int k = std::min(group, nall);
    std::partial_sort(all, all + k, all + nall, detail::match_less);
    for (int i = 0; i < k; ++i) {
        out[i] = all[i];
    }
    return k;
}

}  // namespace nss
