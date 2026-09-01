#include "nss/cpu_api.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

namespace {

struct NlmCase {
    int d;
    int a;
    int s;
    float h;
};

void run_full(const std::vector<std::vector<float>>& frames, int width, int height, int stride, const NlmCase& cfg,
              std::vector<float>& out) {
    const int nc = 1;
    const int extra = cfg.d != 0 ? 1 : 0;
    const int size = height * stride;
    std::vector<float> weight(static_cast<std::size_t>(size), 0.f);
    std::vector<float> wdst(static_cast<std::size_t>(size), 0.f);
    std::vector<float> max_weight(static_cast<std::size_t>(size), std::numeric_limits<float>::epsilon());
    std::vector<float> temp(static_cast<std::size_t>(size), 0.f);
    std::vector<float> temp_bwd(static_cast<std::size_t>(size), 0.f);
    std::vector<float> temp_fwd(static_cast<std::size_t>(size), 0.f);
    std::vector<float> buffer(static_cast<std::size_t>(width), 0.f);
    const float h2_inv_norm =
        (255.0f * 255.0f) / (3.0f * cfg.h * cfg.h * static_cast<float>((2 * cfg.s + 1) * (2 * cfg.s + 1)));
    const int span = 2 * cfg.a + 1;
    const float* ref_center = frames[static_cast<std::size_t>(cfg.d)].data();

    for (int i = -cfg.d; i <= 0; ++i) {
        const int bwd_slot = cfg.d + i;
        const int fwd_slot = cfg.d - i;
        const float* ref_bwd = frames[static_cast<std::size_t>(bwd_slot)].data();
        const float* ref_fwd = frames[static_cast<std::size_t>(fwd_slot)].data();
        const float* src_bwd = ref_bwd;
        const float* src_fwd = ref_fwd;
        for (int oy = -cfg.a; oy <= cfg.a; ++oy) {
            for (int ox = -cfg.a; ox <= cfg.a; ++ox) {
                if (i * span * span + oy * span + ox >= 0) {
                    continue;
                }
                nss::nlm_distance_luma_f32(temp_bwd.data(), ref_center, ref_bwd, ox, oy, width, height, stride);
                nss::nlm_horizontal(temp.data(), temp_bwd.data(), cfg.s, width, height, stride);
                nss::nlm_vertical_welsch(temp_bwd.data(), temp.data(), cfg.s, h2_inv_norm, width, height, stride,
                                          buffer.data());
                if (i == 0) {
                    nss::nlm_accum_ch1(weight.data(), wdst.data(), max_weight.data(), src_bwd, src_bwd,
                                       temp_bwd.data(), temp_bwd.data(), ox, oy, width, height, stride);
                    continue;
                }
                nss::nlm_distance_luma_f32(temp_fwd.data(), ref_fwd, ref_center, ox, oy, width, height, stride);
                nss::nlm_horizontal(temp.data(), temp_fwd.data(), cfg.s, width, height, stride);
                nss::nlm_vertical_welsch(temp_fwd.data(), temp.data(), cfg.s, h2_inv_norm, width, height, stride,
                                          buffer.data());
                nss::nlm_accum_ch1(weight.data(), wdst.data(), max_weight.data(), src_bwd, src_fwd,
                                   temp_bwd.data(), temp_fwd.data(), ox, oy, width, height, stride);
            }
        }
    }
    (void)nc;
    (void)extra;
    out.assign(static_cast<std::size_t>(size), 0.f);
    nss::nlm_finish_ch1(out.data(), ref_center, weight.data(), wdst.data(), max_weight.data(), 1.f, width, height,
                        stride);
}

void run_stripes(const std::vector<std::vector<float>>& frames, int width, int height, int stride, const NlmCase& cfg,
                 int core_rows, std::vector<float>& out) {
    const int halo = cfg.a + cfg.s;
    const float h2_inv_norm =
        (255.0f * 255.0f) / (3.0f * cfg.h * cfg.h * static_cast<float>((2 * cfg.s + 1) * (2 * cfg.s + 1)));
    const float* ref_center = frames[static_cast<std::size_t>(cfg.d)].data();
    out.assign(static_cast<std::size_t>(height * stride), 0.f);

    for (int core0 = 0; core0 < height; core0 += core_rows) {
        const int core1 = std::min(height, core0 + core_rows);
        const int ext0 = std::max(0, core0 - halo);
        const int ext1 = std::min(height, core1 + halo);
        const int ext_h = ext1 - ext0;
        const int ext_size = ext_h * stride;
        std::vector<float> weight(static_cast<std::size_t>(ext_size), 0.f);
        std::vector<float> wdst(static_cast<std::size_t>(ext_size), 0.f);
        std::vector<float> max_weight(static_cast<std::size_t>(ext_size), std::numeric_limits<float>::epsilon());
        std::vector<float> temp(static_cast<std::size_t>(ext_size), 0.f);
        std::vector<float> temp_bwd(static_cast<std::size_t>(ext_size), 0.f);
        std::vector<float> temp_fwd(static_cast<std::size_t>(ext_size), 0.f);
        std::vector<float> buffer(static_cast<std::size_t>(width), 0.f);
        const int span = 2 * cfg.a + 1;

        auto row_ptr = [&](const std::vector<float>& frame) { return frame.data() + ext0 * stride; };
        for (int i = -cfg.d; i <= 0; ++i) {
            const int bwd_slot = cfg.d + i;
            const int fwd_slot = cfg.d - i;
            const float* ref_bwd = row_ptr(frames[static_cast<std::size_t>(bwd_slot)]);
            const float* ref_fwd = row_ptr(frames[static_cast<std::size_t>(fwd_slot)]);
            const float* src_bwd = ref_bwd;
            const float* src_fwd = ref_fwd;
            const float* ref_local = row_ptr(frames[static_cast<std::size_t>(cfg.d)]);
            for (int oy = -cfg.a; oy <= cfg.a; ++oy) {
                for (int ox = -cfg.a; ox <= cfg.a; ++ox) {
                    if (i * span * span + oy * span + ox >= 0) {
                        continue;
                    }
                    nss::nlm_distance_luma_f32(temp_bwd.data(), ref_local, ref_bwd, ox, oy, width, ext_h, stride);
                    nss::nlm_horizontal(temp.data(), temp_bwd.data(), cfg.s, width, ext_h, stride);
                    nss::nlm_vertical_welsch(temp_bwd.data(), temp.data(), cfg.s, h2_inv_norm, width, ext_h, stride,
                                              buffer.data());
                    if (i == 0) {
                        nss::nlm_accum_ch1(weight.data(), wdst.data(), max_weight.data(), src_bwd, src_bwd,
                                           temp_bwd.data(), temp_bwd.data(), ox, oy, width, ext_h, stride);
                        continue;
                    }
                    nss::nlm_distance_luma_f32(temp_fwd.data(), ref_fwd, ref_local, ox, oy, width, ext_h, stride);
                    nss::nlm_horizontal(temp.data(), temp_fwd.data(), cfg.s, width, ext_h, stride);
                    nss::nlm_vertical_welsch(temp_fwd.data(), temp.data(), cfg.s, h2_inv_norm, width, ext_h, stride,
                                              buffer.data());
                    nss::nlm_accum_ch1(weight.data(), wdst.data(), max_weight.data(), src_bwd, src_fwd,
                                       temp_bwd.data(), temp_fwd.data(), ox, oy, width, ext_h, stride);
                }
            }
        }
        const int local = core0 - ext0;
        nss::nlm_finish_ch1(out.data() + core0 * stride, ref_center + core0 * stride,
                            weight.data() + local * stride, wdst.data() + local * stride,
                            max_weight.data() + local * stride, 1.f, width, core1 - core0, stride);
    }
}

int check_case(const NlmCase& cfg) {
    constexpr int width = 19;
    constexpr int height = 23;
    constexpr int stride = 32;
    std::mt19937 rng(8128 + cfg.d * 100 + cfg.s);
    std::uniform_real_distribution<float> random(0.f, 1.f);
    std::vector<std::vector<float>> frames(static_cast<std::size_t>(2 * cfg.d + 1),
                                           std::vector<float>(static_cast<std::size_t>(stride * height), 0.f));
    for (auto& frame : frames) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                frame[static_cast<std::size_t>(y * stride + x)] = random(rng);
            }
        }
    }
    std::vector<float> full;
    run_full(frames, width, height, stride, cfg, full);
    for (int rows : {1, 2, 5, 9, height}) {
        std::vector<float> striped;
        run_stripes(frames, width, height, stride, cfg, rows, striped);
        double max_abs = 0.0;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                max_abs = std::max(max_abs, static_cast<double>(std::fabs(
                                      full[static_cast<std::size_t>(y * stride + x)] -
                                      striped[static_cast<std::size_t>(y * stride + x)])));
            }
        }
        std::printf("nlm stripe d=%d a=%d s=%d rows=%d max_abs=%.6g\n", cfg.d, cfg.a, cfg.s, rows, max_abs);
        if (!(max_abs < 2e-5)) {
            std::fprintf(stderr, "stripe result differs from full-frame result\n");
            return 1;
        }
    }
    return 0;
}

}  // namespace

int main() {
    int failed = 0;
    failed |= check_case({0, 2, 0, 1.2f});
    failed |= check_case({1, 2, 4, 1.2f});
    failed |= check_case({1, 1, 2, 0.8f});
    return failed ? 1 : 0;
}
