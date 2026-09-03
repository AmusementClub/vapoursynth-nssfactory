#include "nss/cpu_api.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

bool check_horizontal() {
    constexpr int h = 3;
    for (const int w : {1, 3, 7, 15, 16, 31, 127, 128, 129, 257}) {
        for (const int padding : {0, 5}) {
            const int stride = w + padding;
            std::vector<float> src(static_cast<std::size_t>(stride * h), -7.f);
            std::vector<float> got(static_cast<std::size_t>(stride * h), -11.f);
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    src[static_cast<std::size_t>(y * stride + x)] =
                        0.0003f * static_cast<float>(1 + (x * 37 + y * 101) % 997);
                }
            }
            for (int radius = 0; radius <= 8; ++radius) {
                std::fill(got.begin(), got.end(), -11.f);
                nss::nlm_horizontal(got.data(), src.data(), radius, w, h, stride);
                for (int y = 0; y < h; ++y) {
                    for (int x = 0; x < w; ++x) {
                        float want = 0.f;
                        for (int j = -radius; j <= radius; ++j) {
                            want += src[static_cast<std::size_t>(y * stride + std::clamp(x + j, 0, w - 1))];
                        }
                        const float value = got[static_cast<std::size_t>(y * stride + x)];
                        const float tolerance = 4e-5f * (1.f + std::fabs(want));
                        if (std::fabs(value - want) > tolerance) {
                            std::fprintf(stderr,
                                         "horizontal mismatch w=%d stride=%d radius=%d at %d,%d: %.9g vs %.9g\n",
                                         w, stride, radius, x, y, static_cast<double>(value),
                                         static_cast<double>(want));
                            return false;
                        }
                    }
                    for (int x = w; x < stride; ++x) {
                        if (got[static_cast<std::size_t>(y * stride + x)] != -11.f) {
                            std::fprintf(stderr, "horizontal overwrote padding w=%d stride=%d radius=%d at %d,%d\n",
                                         w, stride, radius, x, y);
                            return false;
                        }
                    }
                }
            }
        }
    }
    return true;
}

bool check_distance_horizontal_luma() {
    int max_ulp = 0;
    float max_abs = 0.f;
    for (const int w : {1, 3, 7, 15, 16, 31, 65}) {
        for (const int h : {1, 2, 5, 11}) {
            const int stride = w + 7;
            std::vector<float> center(static_cast<std::size_t>(stride * h), -3.f);
            std::vector<float> neighbor(static_cast<std::size_t>(stride * h), -5.f);
            std::vector<float> distance(static_cast<std::size_t>(stride * h), -7.f);
            std::vector<float> expected(static_cast<std::size_t>(stride * h), -11.f);
            std::vector<float> got(static_cast<std::size_t>(stride * h), -11.f);
            std::vector<float> repeat(static_cast<std::size_t>(stride * h), -11.f);
            std::vector<float> scratch(static_cast<std::size_t>(stride), -13.f);
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    center[static_cast<std::size_t>(y * stride + x)] =
                        0.001f * static_cast<float>(1 + (x * 37 + y * 101) % 997);
                    neighbor[static_cast<std::size_t>(y * stride + x)] =
                        -0.0013f * static_cast<float>(1 + (x * 53 + y * 89) % 991);
                }
            }
            for (int radius = 0; radius <= 6; ++radius) {
                for (const int oy : {-2, -1, 0, 1, 2}) {
                    for (const int ox : {-2, -1, 0, 1, 2}) {
                        if (std::abs(ox) >= w) {
                            continue;
                        }
                        std::fill(distance.begin(), distance.end(), -7.f);
                        std::fill(expected.begin(), expected.end(), -11.f);
                        std::fill(got.begin(), got.end(), -11.f);
                        std::fill(repeat.begin(), repeat.end(), -11.f);
                        std::fill(scratch.begin(), scratch.end(), -13.f);
                        nss::nlm_distance_luma_f32(distance.data(), center.data(), neighbor.data(), ox, oy, w, h,
                                                   stride);
                        nss::nlm_horizontal(expected.data(), distance.data(), radius, w, h, stride);
                        nss::nlm_distance_luma_horizontal_f32(got.data(), scratch.data(), center.data(),
                                                              neighbor.data(), ox, oy, radius, w, h, stride);
                        nss::nlm_distance_luma_horizontal_f32(repeat.data(), scratch.data(), center.data(),
                                                              neighbor.data(), ox, oy, radius, w, h, stride);
                        for (int y = 0; y < h; ++y) {
                            for (int x = 0; x < w; ++x) {
                                const float want = expected[static_cast<std::size_t>(y * stride + x)];
                                const float value = got[static_cast<std::size_t>(y * stride + x)];
                                const float abs_error = std::fabs(value - want);
                                const auto want_bits = std::bit_cast<std::uint32_t>(want);
                                const auto value_bits = std::bit_cast<std::uint32_t>(value);
                                const int ulp = static_cast<int>(want_bits > value_bits ? want_bits - value_bits
                                                                                     : value_bits - want_bits);
                                max_abs = std::max(max_abs, abs_error);
                                max_ulp = std::max(max_ulp, ulp);
                                if (!std::isfinite(value) || ulp > 8 ||
                                    abs_error > 2e-6f * (1.f + std::fabs(want))) {
                                    std::fprintf(stderr,
                                                 "distance-horizontal mismatch w=%d h=%d radius=%d offset=%d,%d "
                                                 "at %d,%d: %a vs %a ulp=%d\n",
                                                 w, h, radius, ox, oy, x, y, static_cast<double>(value),
                                                 static_cast<double>(want), ulp);
                                    return false;
                                }
                            }
                            if (std::memcmp(got.data() + static_cast<std::size_t>(y * stride),
                                            repeat.data() + static_cast<std::size_t>(y * stride),
                                            static_cast<std::size_t>(w) * sizeof(float)) != 0) {
                                std::fprintf(stderr, "distance-horizontal is not repeat-exact\n");
                                return false;
                            }
                            for (int x = w; x < stride; ++x) {
                                if (got[static_cast<std::size_t>(y * stride + x)] != -11.f) {
                                    std::fprintf(stderr, "distance-horizontal overwrote padding at %d,%d\n", x, y);
                                    return false;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    std::printf("distance-horizontal differential max_abs=%g max_ulp=%d\n", max_abs, max_ulp);
    return true;
}

bool check_vertical_welsch() {
    for (const float scale : {0.037f, 1000.f}) {
        for (const int w : {1, 3, 7, 16, 31, 65}) {
            for (const int h : {1, 2, 5, 11}) {
                const int stride = w + 5;
                std::vector<float> src(static_cast<std::size_t>(stride * h), -7.f);
                std::vector<float> got(static_cast<std::size_t>(stride * h), -11.f);
                std::vector<float> buffer(static_cast<std::size_t>(w), 0.f);
                for (int y = 0; y < h; ++y) {
                    for (int x = 0; x < w; ++x) {
                        src[static_cast<std::size_t>(y * stride + x)] =
                            0.0002f * static_cast<float>(1 + (x * 37 + y * 101) % 997);
                    }
                }
                for (int radius = 0; radius <= 8; ++radius) {
                    std::fill(got.begin(), got.end(), -11.f);
                    nss::nlm_vertical_welsch(got.data(), src.data(), radius, scale, w, h, stride, buffer.data());
                    for (int y = 0; y < h; ++y) {
                        for (int x = 0; x < w; ++x) {
                            float sum = 0.f;
                            for (int j = -radius; j <= radius; ++j) {
                                sum += src[static_cast<std::size_t>(std::clamp(y + j, 0, h - 1) * stride + x)];
                            }
                            const float want = std::exp(-scale * sum);
                            const float value = got[static_cast<std::size_t>(y * stride + x)];
                            if (std::fabs(value - want) > 8e-6f) {
                                std::fprintf(stderr,
                                             "vertical Welsch mismatch w=%d h=%d radius=%d at %d,%d: %.9g vs %.9g\n",
                                             w, h, radius, x, y, static_cast<double>(value),
                                             static_cast<double>(want));
                                return false;
                            }
                        }
                        for (int x = w; x < stride; ++x) {
                            if (got[static_cast<std::size_t>(y * stride + x)] != -11.f) {
                                std::fprintf(stderr,
                                             "vertical Welsch overwrote padding w=%d h=%d radius=%d at %d,%d\n", w,
                                             h, radius, x, y);
                                return false;
                            }
                        }
                    }
                    if (h > 1) {
                        const int y0 = h / 2;
                        const int rows = h - y0;
                        std::vector<float> compact(static_cast<std::size_t>(rows * stride), -13.f);
                        nss::nlm_vertical_welsch_range(compact.data(), src.data(), radius, scale, w, h, stride, y0, h,
                                                       buffer.data());
                        for (int y = y0; y < h; ++y) {
                            for (int x = 0; x < stride; ++x) {
                                const float value = compact[static_cast<std::size_t>(y - y0) * stride + x];
                                const float want = x < w ? got[static_cast<std::size_t>(y) * stride + x] : -13.f;
                                const bool mismatch = x < w ? std::fabs(value - want) > 8e-6f : value != want;
                                if (mismatch) {
                                    std::fprintf(stderr,
                                                 "vertical Welsch range mismatch w=%d h=%d radius=%d at %d,%d\n", w,
                                                 h, radius, x, y);
                                    return false;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return true;
}

bool check_mixed_plane_strides() {
    constexpr int w = 7;
    constexpr int h = 5;
    constexpr int map_stride = 11;
    const std::array<int, 3> source_strides{13, 17, 19};
    std::array<std::vector<float>, 3> center;
    std::array<std::vector<float>, 3> neighbor;
    for (int c = 0; c < 3; ++c) {
        center[c].resize(static_cast<std::size_t>(source_strides[c] * h));
        neighbor[c].resize(static_cast<std::size_t>(source_strides[c] * h));
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < source_strides[c]; ++x) {
                center[c][static_cast<std::size_t>(y * source_strides[c] + x)] =
                    0.1f * static_cast<float>(c + 1) + 0.01f * static_cast<float>(x + 3 * y);
                neighbor[c][static_cast<std::size_t>(y * source_strides[c] + x)] =
                    -0.07f * static_cast<float>(c + 1) + 0.015f * static_cast<float>(2 * x + y);
            }
        }
    }
    std::array<const float*, 3> cp{center[0].data(), center[1].data(), center[2].data()};
    std::array<const float*, 3> np{neighbor[0].data(), neighbor[1].data(), neighbor[2].data()};

    for (const nss::ChannelMode mode : {nss::ChannelMode::Y, nss::ChannelMode::UV, nss::ChannelMode::YUV,
                                        nss::ChannelMode::RGB}) {
        std::vector<float> got(static_cast<std::size_t>(map_stride * h), -1.f);
        nss::nlm_distance_strided_f32(got.data(), cp.data(), source_strides.data(), np.data(), source_strides.data(),
                                      mode, -2, 1, w, h, map_stride);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const int nx = std::max(0, std::min(w - 1, x - 2));
                const int ny = std::max(0, std::min(h - 1, y + 1));
                float expected = 0.f;
                if (mode == nss::ChannelMode::Y) {
                    const float d = center[0][y * source_strides[0] + x] - neighbor[0][ny * source_strides[0] + nx];
                    expected = 3.f * d * d;
                } else if (mode == nss::ChannelMode::UV) {
                    for (int c = 1; c < 3; ++c) {
                        const float d = center[c][y * source_strides[c] + x] - neighbor[c][ny * source_strides[c] + nx];
                        expected += d * d;
                    }
                    expected *= 1.5f;
                } else if (mode == nss::ChannelMode::YUV) {
                    for (int c = 0; c < 3; ++c) {
                        const float d = center[c][y * source_strides[c] + x] - neighbor[c][ny * source_strides[c] + nx];
                        expected += d * d;
                    }
                } else {
                    const float u1 = center[0][y * source_strides[0] + x];
                    const float v1 = neighbor[0][ny * source_strides[0] + nx];
                    const float u2 = center[1][y * source_strides[1] + x];
                    const float v2 = neighbor[1][ny * source_strides[1] + nx];
                    const float u3 = center[2][y * source_strides[2] + x];
                    const float v3 = neighbor[2][ny * source_strides[2] + nx];
                    const float mr = (u1 + v1) / 6.f;
                    expected = (2.f / 3.f + mr) * (u1 - v1) * (u1 - v1) + (4.f / 3.f) * (u2 - v2) * (u2 - v2) +
                              (1.f - mr) * (u3 - v3) * (u3 - v3);
                }
                if (std::fabs(got[static_cast<std::size_t>(y * map_stride + x)] - expected) > 2e-6f) {
                    std::fprintf(stderr, "mixed-stride distance mismatch mode=%d at %d,%d\n", static_cast<int>(mode), x,
                                 y);
                    return false;
                }
            }
        }
    }

    std::array<std::vector<float>, 3> sb;
    std::array<std::vector<float>, 3> sf;
    for (int c = 0; c < 3; ++c) {
        sb[c].resize(static_cast<std::size_t>(source_strides[c] * h));
        sf[c].resize(static_cast<std::size_t>(source_strides[c] * h));
        for (std::size_t i = 0; i < sb[c].size(); ++i) {
            sb[c][i] = 0.02f * static_cast<float>(i + c + 1);
            sf[c][i] = -0.015f * static_cast<float>(i + 2 * c + 1);
        }
    }
    std::array<const float*, 3> sbp{sb[0].data(), sb[1].data(), sb[2].data()};
    std::array<const float*, 3> sfp{sf[0].data(), sf[1].data(), sf[2].data()};
    std::vector<float> temp1(static_cast<std::size_t>(map_stride * h));
    std::vector<float> temp2(static_cast<std::size_t>(map_stride * h));
    for (std::size_t i = 0; i < temp1.size(); ++i) {
        temp1[i] = 0.01f * static_cast<float>(i + 1);
        temp2[i] = 0.013f * static_cast<float>(i + 2);
    }
    std::vector<float> weight(static_cast<std::size_t>(map_stride * 3), 0.2f);
    std::vector<float> maxw(static_cast<std::size_t>(map_stride * 3), 0.4f);
    std::array<std::vector<float>, 3> wdst;
    for (auto& v : wdst) {
        v.assign(static_cast<std::size_t>(map_stride * 3), -0.1f);
    }
    auto expected_weight = weight;
    auto expected_max = maxw;
    auto expected_wdst = wdst;
    constexpr int y0 = 1;
    constexpr int y1 = 4;
    constexpr int ox = -1;
    constexpr int oy = 2;
    for (int y = y0; y < y1; ++y) {
        for (int x = 0; x < w; ++x) {
            const int out = (y - y0) * map_stride + x;
            const int mqy = std::max(0, std::min(h - 1, y - oy));
            const int pqy = std::max(0, std::min(h - 1, y + oy));
            const int mqx = std::max(0, std::min(w - 1, x - ox));
            const int pqx = std::max(0, std::min(w - 1, x + ox));
            const float u = temp1[y * map_stride + x];
            const float um = temp2[mqy * map_stride + mqx];
            expected_weight[out] += u + um;
            expected_max[out] = std::max(expected_max[out], std::max(u, um));
            for (int c = 0; c < 3; ++c) {
                expected_wdst[c][out] += u * sb[c][pqy * source_strides[c] + pqx] +
                                         um * sf[c][mqy * source_strides[c] + mqx];
            }
        }
    }
    std::array<float*, 3> wdp{wdst[0].data(), wdst[1].data(), wdst[2].data()};
    nss::nlm_accum_strided(weight.data(), wdp[0], wdp[1], wdp[2], maxw.data(), sbp.data(), source_strides.data(),
                           sfp.data(), source_strides.data(), temp1.data(), temp2.data(), 3, ox, oy, w, h, map_stride,
                           y0, y1, 0);
    for (std::size_t i = 0; i < weight.size(); ++i) {
        if (std::fabs(weight[i] - expected_weight[i]) > 2e-6f || std::fabs(maxw[i] - expected_max[i]) > 2e-6f) {
            std::fprintf(stderr, "mixed-stride accumulation state mismatch at %zu\n", i);
            return false;
        }
        for (int c = 0; c < 3; ++c) {
            if (std::fabs(wdst[c][i] - expected_wdst[c][i]) > 2e-6f) {
                std::fprintf(stderr, "mixed-stride accumulation channel mismatch at %zu\n", i);
                return false;
            }
        }
    }

    const std::array<int, 3> dst_strides{15, 17, 21};
    std::array<std::vector<float>, 3> src_core;
    std::array<std::vector<float>, 3> dst;
    std::array<const float*, 3> srcp{};
    std::array<float*, 3> dstp{};
    std::array<const float*, 3> wdst_const{wdst[0].data(), wdst[1].data(), wdst[2].data()};
    std::vector<float> finish_w(static_cast<std::size_t>(map_stride * 3), 0.7f);
    std::vector<float> finish_m(static_cast<std::size_t>(map_stride * 3), 1.1f);
    for (int c = 0; c < 3; ++c) {
        src_core[c].resize(static_cast<std::size_t>(source_strides[c] * 3));
        dst[c].assign(static_cast<std::size_t>(dst_strides[c] * 3), -9.f);
        for (std::size_t i = 0; i < src_core[c].size(); ++i) {
            src_core[c][i] = 0.03f * static_cast<float>(i + c + 1);
        }
        srcp[c] = src_core[c].data();
        dstp[c] = dst[c].data();
    }
    nss::nlm_finish_strided(dstp.data(), dst_strides.data(), srcp.data(), source_strides.data(), finish_w.data(),
                            wdst_const.data(), finish_m.data(), 0.8f, 3, w, 3, map_stride);
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < w; ++x) {
            const int map = y * map_stride + x;
            const float mul = 0.8f * finish_m[map];
            for (int c = 0; c < 3; ++c) {
                const float expected = (mul * src_core[c][y * source_strides[c] + x] + wdst[c][map]) /
                                       (mul + finish_w[map]);
                if (std::fabs(dst[c][y * dst_strides[c] + x] - expected) > 2e-6f) {
                    std::fprintf(stderr, "mixed-stride finish mismatch at c=%d,%d,%d\n", c, x, y);
                    return false;
                }
            }
        }
    }
    return true;
}

}  // namespace

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
    if (!check_horizontal() || !check_distance_horizontal_luma() || !check_vertical_welsch() ||
        !check_mixed_plane_strides()) {
        return 1;
    }
    return 0;
}
