#include "nss/cpu_api.hpp"
#include "cpu/bm/matcher.hpp"

#include <algorithm>
#include <cstdint>

namespace nss {

int predictive_match(const float* const* refs, const int* strides, int ntemp, int width, int height, int bx, int by,
                     int t0, const SearchConfig& cfg, Match* out, unsigned avx2_features) {
    if (!refs || !strides || !out || ntemp < 1 || t0 < 0 || t0 >= ntemp || width < cfg.block || height < cfg.block || cfg.block < 1 ||
        cfg.group < 1 || cfg.group > kBmMaxGroup || cfg.step < 1 || cfg.bm_range < 0 || cfg.ps_num < 1 ||
        cfg.ps_range < 0 || cfg.radius < 0 || cfg.radius > kBmMaxRadius || !refs[t0] || strides[t0] < width) {
        return 0;
    }
    if (cfg.valid_t_begin < 0 || cfg.valid_t_begin > t0 ||
        (cfg.valid_t_end != -1 && (cfg.valid_t_end <= t0 || cfg.valid_t_end > ntemp))) return 0;
    for (int t = 0; t < ntemp; ++t) {
        if (!refs[t] || strides[t] < width) {
            return 0;
        }
    }
    const int block = cfg.block;
    const int group = cfg.group;
    int n = spatial_match(refs[t0], strides[t0], width, height, bx, by, block, cfg.bm_range, group, out, avx2_features);
    const int cx = std::clamp(bx, 0, width - block);
    const int cy = std::clamp(by, 0, height - block);
    const float* reference = refs[t0] + cy * strides[t0] + cx;
    return detail::collect_temporal(width, height, t0, ntemp, cfg, out, n, [&](int t, int x, int y) {
        return ssd_block(reference, strides[t0], refs[t] + y * strides[t] + x, strides[t], block);
    });
}

}  // namespace nss
