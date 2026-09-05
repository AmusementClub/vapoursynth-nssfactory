#include "nss/cpu_api.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int roundtrip_2d(int n, double limit) {
    std::mt19937 rng(7 + n);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    const int m = n * n;
    std::vector<float> orig(static_cast<std::size_t>(m));
    std::vector<float> block(static_cast<std::size_t>(m));
    double max_err = 0.0;
    double nrm = 0.0;
    for (int i = 0; i < m; ++i) {
        orig[static_cast<std::size_t>(i)] = dist(rng);
        block[static_cast<std::size_t>(i)] = orig[static_cast<std::size_t>(i)];
        nrm += static_cast<double>(orig[static_cast<std::size_t>(i)]) * orig[static_cast<std::size_t>(i)];
    }
    nss::dct_2d(block.data(), n);
    nss::idct_2d(block.data(), n);
    for (int i = 0; i < m; ++i) {
        const double e = std::fabs(static_cast<double>(block[static_cast<std::size_t>(i)] - orig[static_cast<std::size_t>(i)]));
        if (e > max_err) {
            max_err = e;
        }
    }
    nrm = std::sqrt(nrm);
    std::printf("dct_2d n=%d max_abs=%.6g rel=%.6g\n", n, max_err, max_err / (nrm + 1e-12));
    if (!(max_err < limit)) {
        std::fprintf(stderr, "dct_2d n=%d roundtrip failed\n", n);
        return 1;
    }
    return 0;
}

static int roundtrip_1d(int n, double limit) {
    std::mt19937 rng(11 + n);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> orig(static_cast<std::size_t>(n));
    std::vector<float> block(static_cast<std::size_t>(n));
    double max_err = 0.0;
    for (int i = 0; i < n; ++i) {
        orig[static_cast<std::size_t>(i)] = dist(rng);
        block[static_cast<std::size_t>(i)] = orig[static_cast<std::size_t>(i)];
    }
    nss::dct_1d(block.data(), block.data(), n);
    nss::idct_1d(block.data(), block.data(), n);
    for (int i = 0; i < n; ++i) {
        const double e = std::fabs(static_cast<double>(block[static_cast<std::size_t>(i)] - orig[static_cast<std::size_t>(i)]));
        if (e > max_err) {
            max_err = e;
        }
    }
    std::printf("dct_1d n=%d max_abs=%.6g\n", n, max_err);
    if (!(max_err < limit)) {
        std::fprintf(stderr, "dct_1d n=%d roundtrip failed\n", n);
        return 1;
    }
    return 0;
}

static int dc_energy() {
    float in[8];
    float out[8];
    for (int i = 0; i < 8; ++i) {
        in[i] = 1.f;
    }
    nss::dct8_1d(in, out);
    const double dc = std::fabs(static_cast<double>(out[0]) - std::sqrt(8.0));
    double ac = 0.0;
    for (int i = 1; i < 8; ++i) {
        ac = std::max(ac, std::fabs(static_cast<double>(out[i])));
    }
    std::printf("dct8 ones dc_err=%.6g ac_max=%.6g\n", dc, ac);
    if (dc > 1e-5 || ac > 1e-5) {
        std::fprintf(stderr, "dct8 DC test failed\n");
        return 1;
    }
    return 0;
}

static void reference_dct(const float* in, float* out, int n) {
    constexpr double pi = 3.14159265358979323846264338327950288;
    for (int k = 0; k < n; ++k) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += static_cast<double>(in[i]) *
                   std::cos(pi * (static_cast<double>(i) + 0.5) * static_cast<double>(k) /
                            static_cast<double>(n));
        }
        const double scale = std::sqrt((k == 0 ? 1.0 : 2.0) / static_cast<double>(n));
        out[k] = static_cast<float>(sum * scale);
    }
}

static int lines_vs_reference(int n, int count, int line_stride, int sample_stride) {
    std::mt19937 rng(99 + n * 101 + count * 7 + line_stride + sample_stride);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    const std::size_t span = static_cast<std::size_t>((count - 1) * line_stride + (n - 1) * sample_stride + 1);
    std::vector<float> rows(span);
    std::vector<float> ref(span);
    std::vector<float> line(static_cast<std::size_t>(n));
    std::vector<float> transformed(static_cast<std::size_t>(n));
    for (float& value : rows) {
        value = dist(rng);
    }
    ref = rows;
    for (int v = 0; v < count; ++v) {
        for (int i = 0; i < n; ++i) {
            line[static_cast<std::size_t>(i)] = rows[static_cast<std::size_t>(v * line_stride + i * sample_stride)];
        }
        reference_dct(line.data(), transformed.data(), n);
        for (int i = 0; i < n; ++i) {
            ref[static_cast<std::size_t>(v * line_stride + i * sample_stride)] =
                transformed[static_cast<std::size_t>(i)];
        }
    }
    std::vector<float> got = rows;
    nss::dct_lines(got.data(), n, line_stride, sample_stride, count, false);
    double max_err = 0.0;
    for (int v = 0; v < count; ++v) {
        for (int i = 0; i < n; ++i) {
            const std::size_t index = static_cast<std::size_t>(v * line_stride + i * sample_stride);
            max_err = std::max(max_err, std::fabs(static_cast<double>(got[index] - ref[index])));
        }
    }
    nss::dct_lines(got.data(), n, line_stride, sample_stride, count, true);
    double rt = 0.0;
    for (int v = 0; v < count; ++v) {
        for (int i = 0; i < n; ++i) {
            const std::size_t index = static_cast<std::size_t>(v * line_stride + i * sample_stride);
            rt = std::max(rt, std::fabs(static_cast<double>(got[index] - rows[index])));
        }
    }
    std::printf("dct_lines n=%d count=%d strides=%d/%d vs_ref=%.6g roundtrip=%.6g\n", n, count,
                line_stride, sample_stride, max_err, rt);
    if (max_err > 2e-5 || rt > 2e-4) {
        std::fprintf(stderr, "dct_lines n=%d count=%d strides=%d/%d failed\n", n, count, line_stride,
                     sample_stride);
        return 1;
    }
    return 0;
}

static int bm3d_sigma0_roundtrip(int block) {
    const int group = 8;
    const int area = block * block;
    std::vector<float> patches(static_cast<std::size_t>(group * area));
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& v : patches) {
        v = dist(rng);
    }
    std::vector<float> orig = patches;
    std::vector<float> work(static_cast<std::size_t>(nss::bm3d_filter_work_floats(group, block)));
    float w = 0.f;
    nss::bm3d_filter_group(patches.data(), area, group, group, block, 0.f, false, nullptr, &w, work.data());
    double max_err = 0.0;
    for (std::size_t i = 0; i < patches.size(); ++i) {
        max_err = std::max(max_err, std::fabs(static_cast<double>(patches[i] - orig[i])));
    }
    std::printf("bm3d sigma0 roundtrip block=%d max_abs=%.6g w=%.6g\n", block, max_err,
                static_cast<double>(w));
    if (max_err > 5e-4) {
        std::fprintf(stderr, "bm3d sigma0 roundtrip failed\n");
        return 1;
    }
    return 0;
}

static int bm3d_direct_matches_group() {
    constexpr int block = 4;
    constexpr int group = 8;
    constexpr int width = 48;
    constexpr int height = 40;
    constexpr int stride = 48;
    constexpr int area = block * block;
    std::vector<float> src(static_cast<std::size_t>(stride * height));
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> dist(-0.4f, 0.4f);
    for (float& v : src) {
        v = dist(rng);
    }
    nss::Match matches[nss::kBmMaxGroup]{};
    const int k = nss::spatial_match(src.data(), stride, width, height, 12, 10, block, 7, group, matches);
    if (k < 1) {
        std::fprintf(stderr, "bm3d_direct matcher returned %d\n", k);
        return 1;
    }
    std::vector<float> packed(static_cast<std::size_t>(group) * area, 0.f);
    for (int g = 0; g < k; ++g) {
        nss::pack_patch(packed.data() + static_cast<std::size_t>(g) * area, area, src.data(), stride, matches[g].x,
                        matches[g].y, block, width, height);
    }
    std::vector<float> work(static_cast<std::size_t>(nss::bm3d_filter_work_floats(group, block)));
    float weight = 1.f;
    nss::bm3d_filter_group(packed.data(), area, group, k, block, 0.02f, false, nullptr, &weight, work.data());
    std::vector<float> num_a(static_cast<std::size_t>(stride * height), 0.f);
    std::vector<float> den_a(num_a.size(), 0.f);
    for (int g = 0; g < k; ++g) {
        nss::aggregate_add(num_a.data(), den_a.data(), stride, matches[g].x, matches[g].y,
                           packed.data() + static_cast<std::size_t>(g) * area, block, width, height, weight);
    }

    std::vector<float> cube(static_cast<std::size_t>(group) * area, 0.f);
    std::vector<float> num_b(num_a.size(), 0.f);
    std::vector<float> den_b(den_a.size(), 0.f);
    nss::bm3d_filter_direct(src.data(), stride, matches, k, block, group, 0.02f, false, nullptr, stride, num_b.data(),
                            den_b.data(), stride, width, height, cube.data(), work.data());
    double max_err = 0.0;
    for (std::size_t i = 0; i < num_a.size(); ++i) {
        max_err = std::max(max_err, std::fabs(static_cast<double>(num_a[i] - num_b[i])));
        max_err = std::max(max_err, std::fabs(static_cast<double>(den_a[i] - den_b[i])));
    }
    std::printf("bm3d_direct vs group block=4 max_abs=%.6g\n", max_err);
    if (max_err > 1e-6) {
        std::fprintf(stderr, "bm3d_direct mismatch\n");
        return 1;
    }
    return 0;
}

// The fused Wiener route reads two independently-strided planes. Compare its
// logical output with tightly-packed storage and retain output-row sentinels.
static int bm3d_independent_strides() {
    constexpr int width = 43, height = 37;
    constexpr int sstride = 49, rstride = 53, dstride = 57;
    constexpr float sentinel = -12345.f;
    std::vector<float> src(width * height), ref(src.size());
    std::vector<float> padded_src(sstride * height + 1, sentinel);
    std::vector<float> padded_ref(rstride * height + 1, sentinel);
    std::mt19937 rng(701);
    std::uniform_real_distribution<float> dist(-.5f, .5f);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            src[y * width + x] = padded_src[1 + y * sstride + x] = dist(rng);
            ref[y * width + x] = padded_ref[1 + y * rstride + x] = dist(rng);
        }
    }
    int failed = 0, cases = 0;
    for (int block : {4, 8, 12, 16}) {
        for (int group : {1, 2, 8, 16, 64}) {
            for (int k : {1, group}) {
                nss::Match matches[nss::kBmMaxGroup]{};
                for (int i = 0; i < k; ++i) {
                    matches[i].x = (i * 7) % (width - block + 1);
                    matches[i].y = (i * 11) % (height - block + 1);
                }
                for (bool wiener : {false, true}) {
                    for (bool fused : {false, true}) {
                        if (fused && (block != 8 || group != 8)) continue;
                        std::vector<float> num(src.size(), 0.f), den(src.size(), 0.f);
                        std::vector<float> padded_num(dstride * height + 2, sentinel), padded_den = padded_num;
                        for (int y = 0; y < height; ++y) {
                            std::fill_n(padded_num.data() + 1 + y * dstride, width, 0.f);
                            std::fill_n(padded_den.data() + 1 + y * dstride, width, 0.f);
                        }
                        std::vector<float> cube(2 * group * block * block);
                        std::vector<float> work(nss::bm3d_filter_work_floats(group, block));
                        auto render = [&](const float* s, int ss, const float* r, int rs, float* n, float* d, int ds) {
                            if (fused) {
                                nss::bm3d_filter8(s, ss, matches, k, .02f, wiener,
                                                  wiener ? r : nullptr, rs, n, d, ds, width, height);
                            } else {
                                nss::bm3d_filter_direct(s, ss, matches, k, block, group, .02f, wiener,
                                                        wiener ? r : nullptr, rs, n, d, ds, width, height,
                                                        cube.data(), work.data());
                            }
                        };
                        render(src.data(), width, ref.data(), width, num.data(), den.data(), width);
                        render(padded_src.data() + 1, sstride, padded_ref.data() + 1, rstride,
                               padded_num.data() + 1, padded_den.data() + 1, dstride);
                        ++cases;
                        for (std::size_t i = 0; i < padded_num.size(); ++i) {
                            const bool valid = i >= 1 && i < 1 + dstride * height && (i - 1) % dstride < width;
                            if (valid) {
                                const std::size_t at = (i - 1) / dstride * width + (i - 1) % dstride;
                                if (std::memcmp(&num[at], &padded_num[i], sizeof(float)) ||
                                    std::memcmp(&den[at], &padded_den[i], sizeof(float))) ++failed;
                            } else if (padded_num[i] != sentinel || padded_den[i] != sentinel) ++failed;
                        }
                    }
                }
            }
        }
    }
    std::printf("bm3d independent strides cases=%d failures=%d\n", cases, failed);
    return failed != 0;
}

int main() {
    const int sizes1d[] = {1, 2, 4, 8, 12, 16, 32, 64};
    const int sizes2d[] = {1, 2, 4, 8, 12, 16, 32};
    const int line_sizes[] = {8, 12, 16, 32, 64};
    const int counts[] = {1, 7, 8, 9, 15, 16, 17};
    int failed = 0;
    failed |= dc_energy();
    for (int n : line_sizes) {
        for (int count : counts) {
            failed |= lines_vs_reference(n, count, n, 1);
            failed |= lines_vs_reference(n, count, 2 * n, 2);
            if (count <= n) {
                failed |= lines_vs_reference(n, count, 1, n);
            }
        }
    }
    failed |= bm3d_sigma0_roundtrip(8);
    failed |= bm3d_sigma0_roundtrip(12);
    failed |= bm3d_direct_matches_group();
    failed |= bm3d_independent_strides();
    for (int n : sizes1d) {
        failed |= roundtrip_1d(n, n >= 32 ? 2e-4 : 1e-4);
    }
    for (int n : sizes2d) {
        failed |= roundtrip_2d(n, n >= 16 ? 5e-4 : 1e-4);
    }
    float block[64];
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (int i = 0; i < 64; ++i) {
        block[i] = dist(rng);
    }
    float copy[64];
    std::memcpy(copy, block, sizeof(copy));
    nss::dct8_2d(block);
    nss::dct_2d(copy, 8);
    double wrap_err = 0.0;
    for (int i = 0; i < 64; ++i) {
        wrap_err = std::max(wrap_err, std::fabs(static_cast<double>(block[i] - copy[i])));
    }
    std::printf("dct8 wrapper vs dct_2d max_abs=%.6g\n", wrap_err);
    if (wrap_err > 1e-6) {
        std::fprintf(stderr, "dct8 wrapper mismatch\n");
        failed = 1;
    }
    return failed ? 1 : 0;
}
