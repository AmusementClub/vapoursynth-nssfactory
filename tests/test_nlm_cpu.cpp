#include "nss/cpu_api.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

int main() {
    constexpr int w = 32;
    constexpr int h = 32;
    constexpr int stride = 32;
    std::vector<float> src(static_cast<std::size_t>(stride * h), 0.4f);
    std::vector<float> dst(static_cast<std::size_t>(stride * h), 0.f);
    std::vector<float> dist(static_cast<std::size_t>(stride * h), 0.f);
    std::vector<float> tmp(static_cast<std::size_t>(stride * h), 0.f);
    std::vector<float> wgt(static_cast<std::size_t>(stride * h), 0.f);
    std::vector<float> wdst(static_cast<std::size_t>(stride * h), 0.f);
    std::vector<float> maxw(static_cast<std::size_t>(stride * h), std::numeric_limits<float>::epsilon());
    std::vector<float> buf(static_cast<std::size_t>(w), 0.f);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            src[static_cast<std::size_t>(y * stride + x)] = 0.25f + 0.01f * static_cast<float>(x + y);
        }
    }

    float n8 = 0.f;
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            const float t = src[static_cast<std::size_t>(y * stride + x)] -
                            src[static_cast<std::size_t>(y * stride + x + 1)];
            n8 += t * t;
        }
    }
    const float ssd8 = nss::ssd_block(src.data(), stride, src.data() + 1, stride, 8);
    if (std::fabs(static_cast<double>(ssd8 - n8)) > 1e-4) {
        std::fprintf(stderr, "ssd_block n=8 mismatch\n");
        return 1;
    }
    float n4 = 0.f;
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            const float t = src[static_cast<std::size_t>(y * stride + x)] -
                            src[static_cast<std::size_t>(y * stride + x + 8)];
            n4 += t * t;
        }
    }
    const float ssd4 = nss::ssd_block(src.data(), stride, src.data() + 8, stride, 4);
    if (std::fabs(static_cast<double>(ssd4 - n4)) > 1e-4) {
        std::fprintf(stderr, "ssd_block n=4 mismatch\n");
        return 1;
    }

    nss::nlm_distance_luma_f32(dist.data(), src.data(), src.data(), 1, 0, w, h, stride);
    nss::nlm_horizontal(tmp.data(), dist.data(), 1, w, h, stride);
    const float h2_inv_norm = (255.0f * 255.0f) / (3.0f * 1.2f * 1.2f * 9.0f);
    nss::nlm_vertical_welsch(dist.data(), tmp.data(), 1, h2_inv_norm, w, h, stride, buf.data());
    nss::nlm_accum_ch1(wgt.data(), wdst.data(), maxw.data(), src.data(), src.data(), dist.data(), dist.data(), 1, 0, w,
                       h, stride);
    nss::nlm_finish_ch1(dst.data(), src.data(), wgt.data(), wdst.data(), maxw.data(), 1.0f, w, h, stride);

    int nan = 0;
    double max_abs = 0.0;
    for (int y = 2; y < h - 2; ++y) {
        for (int x = 2; x < w - 2; ++x) {
            const float v = dst[static_cast<std::size_t>(y * stride + x)];
            if (!std::isfinite(v)) {
                ++nan;
            }
            const float s = src[static_cast<std::size_t>(y * stride + x)];
            max_abs = std::max(max_abs, static_cast<double>(std::fabs(v - s)));
        }
    }
    std::printf("nlm_cpu finite=%d max_abs_vs_src=%.6g\n", nan == 0, max_abs);
    if (nan != 0) {
        std::fprintf(stderr, "nlm produced non-finite values\n");
        return 1;
    }
    return 0;
}
