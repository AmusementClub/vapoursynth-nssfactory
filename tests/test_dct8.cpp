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

static int lines_vs_1d(int count) {
    std::mt19937 rng(99 + count);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> rows(static_cast<std::size_t>(count) * 8);
    std::vector<float> ref(rows.size());
    for (int v = 0; v < count; ++v) {
        for (int i = 0; i < 8; ++i) {
            rows[static_cast<std::size_t>(v * 8 + i)] = dist(rng);
        }
        nss::dct8_1d(rows.data() + v * 8, ref.data() + v * 8);
    }
    std::vector<float> got = rows;
    nss::dct_lines(got.data(), 8, 8, 1, count, false);
    double max_err = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        max_err = std::max(max_err, std::fabs(static_cast<double>(got[i] - ref[i])));
    }
    nss::dct_lines(got.data(), 8, 8, 1, count, true);
    double rt = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        rt = std::max(rt, std::fabs(static_cast<double>(got[i] - rows[i])));
    }
    std::printf("dct_lines count=%d vs_1d=%.6g roundtrip=%.6g\n", count, max_err, rt);
    if (max_err > 2e-5 || rt > 2e-4) {
        std::fprintf(stderr, "dct_lines count=%d failed\n", count);
        return 1;
    }
    return 0;
}

static int bm3d_sigma0_roundtrip() {
    const int group = 8;
    const int block = 8;
    const int area = 64;
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
    std::printf("bm3d sigma0 roundtrip max_abs=%.6g w=%.6g\n", max_err, static_cast<double>(w));
    if (max_err > 5e-4) {
        std::fprintf(stderr, "bm3d sigma0 roundtrip failed\n");
        return 1;
    }
    return 0;
}

int main() {
    const int sizes1d[] = {1, 2, 4, 8, 16, 32, 64};
    const int sizes2d[] = {1, 2, 4, 8, 16, 32};
    int failed = 0;
    failed |= dc_energy();
    failed |= lines_vs_1d(8);
    failed |= lines_vs_1d(16);
    failed |= lines_vs_1d(64);
    failed |= bm3d_sigma0_roundtrip();
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
