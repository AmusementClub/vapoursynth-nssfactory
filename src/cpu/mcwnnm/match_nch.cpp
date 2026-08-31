#include "nss/cpu_api.hpp"
#include "nss/cpu_mcwnnm.hpp"

#include <algorithm>
#include <cmath>

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
    if (!refs || !strides || !out || nch < 1 || block < 1 || group < 1) {
        return 0;
    }
    const int max_x = width - block;
    const int max_y = height - block;
    if (max_x < 0 || max_y < 0) {
        return 0;
    }
    const int cx = std::clamp(bx, 0, max_x);
    const int cy = std::clamp(by, 0, max_y);
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
        } else if (dist < out[worst].dist) {
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
    for (int y = top; y <= bottom; ++y) {
        int x = left;
        if (block == 8) {
            for (; x + 1 <= right; x += 2) {
                const float abort_at = (n >= group) ? out[worst].dist : -1.f;
                float d0 = 0.f;
                float d1 = 0.f;
                bool skip0 = false;
                bool skip1 = false;
                for (int c = 0; c < nch; ++c) {
                    if (!refs[c]) {
                        continue;
                    }
                    const float* self = refs[c] + cy * strides[c] + cx;
                    const float* row = refs[c] + y * strides[c];
                    if (!skip0) {
                        d0 += ssd_block(self, strides[c], row + x, strides[c], 8);
                        if (abort_at >= 0.f && d0 >= abort_at) {
                            skip0 = true;
                        }
                    }
                    if (!skip1) {
                        d1 += ssd_block(self, strides[c], row + x + 1, strides[c], 8);
                        if (abort_at >= 0.f && d1 >= abort_at) {
                            skip1 = true;
                        }
                    }
                    if (skip0 && skip1) {
                        break;
                    }
                }
                if (!skip0) {
                    consider(x, y, d0);
                }
                if (!skip1) {
                    consider(x + 1, y, d1);
                }
            }
        }
        for (; x <= right; ++x) {
            consider(x, y, DistNch(refs, strides, nch, cx, cy, x, y, block));
        }
    }
    return n;
}

int predictive_match_nch(const float* const* refs, const int* strides, int nch, int ntemp, int width, int height,
                         int bx, int by, int t0, const SearchConfig& cfg, Match* out) {
    if (!refs || !strides || !out || nch < 1 || ntemp < 1) {
        return 0;
    }
    const float* cur[3];
    int cur_st[3];
    const int nch_use = std::min(nch, 3);
    for (int c = 0; c < nch_use; ++c) {
        cur[c] = refs[c * ntemp + t0];
        cur_st[c] = strides[c];
    }
    int n = spatial_match_nch(cur, cur_st, nch_use, width, height, bx, by, cfg.block, cfg.bm_range, cfg.group, out);
    for (int i = 0; i < n; ++i) {
        out[i].t = t0;
    }
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
    const int k = std::min(cfg.group, nall);
    std::partial_sort(all, all + k, all + nall, [](const Match& a, const Match& b) {
        if (a.dist != b.dist) {
            return a.dist < b.dist;
        }
        return a.t < b.t;
    });
    for (int i = 0; i < k; ++i) {
        out[i] = all[i];
    }
    return k;
}

}  // namespace nss
