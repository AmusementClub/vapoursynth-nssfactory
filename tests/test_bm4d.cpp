// SPDX-License-Identifier: GPL-2.0-only
// BM4D kernel: tile geometry, a constant volume, denoising of a moving
// pattern, small volumes, and the same bits from every SIMD target.
#include "nss/cpu_bm4d.hpp"
#include "nss/params/bm4d.hpp"
#include "nss/backend.hpp"
#include "hwy/targets.h"
#include "hq_test_target.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

struct Gauss {  // Irwin-Hall, platform independent
    std::mt19937 rng;
    explicit Gauss(std::uint32_t seed) : rng(seed) {}
    float operator()() {
        float s = 0.f;
        for (int i = 0; i < 12; ++i) s += static_cast<float>(rng() >> 8) * (1.f / 16777216.f);
        return s - 6.f;
    }
};

struct Video {
    int t, h, w;
    std::vector<float> clean, noisy, out;
};

Video moving(int t, int h, int w, float sigma, std::uint32_t seed) {
    Video v{t, h, w, {}, {}, {}};
    const std::size_t n = static_cast<std::size_t>(t) * h * w;
    v.clean.resize(n);
    v.noisy.resize(n);
    v.out.resize(n);
    Gauss g(seed);
    for (int z = 0; z < t; ++z)
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const int xs = x + z;  // one pixel a frame
                const int tri = (xs * 2 + y * 3) % 40;
                float c = 0.3f + 0.3f * static_cast<float>(((xs / 9) + (y / 7)) % 2) + 0.004f * static_cast<float>(tri < 20 ? tri : 40 - tri);
                if ((xs - 30) * (xs - 30) + (y - 20) * (y - 20) < 100) c = 0.85f;
                const std::size_t i = (static_cast<std::size_t>(z) * h + y) * w + x;
                v.clean[i] = c;
                v.noisy[i] = c + sigma * g();
            }
    return v;
}

void run(Video& v, float sigma, std::vector<float>* basic = nullptr) {
    const std::size_t need = nss::bm4d_work_floats(v.t, v.h, v.w);
    std::vector<float> work(need + 16);
    if (basic) basic->resize(v.out.size());
    nss::bm4d_denoise_volume(v.noisy.data(), v.t, v.h, v.w, sigma, v.out.data(), work.data(), need,
                             basic ? basic->data() : nullptr);
}

double mse(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0;
    for (std::size_t i = 0; i < a.size(); ++i) s += (double(a[i]) - b[i]) * (double(a[i]) - b[i]);
    return s / a.size();
}

void test_geometry() {
    expect(nss::bm4d_tile_count(1920) == 8 && nss::bm4d_tile_count(1080) == 4 && nss::bm4d_tile_count(100) == 1,
           "tile counts");
    expect(nss::bm4d_chunk_first(0) == 0 && nss::bm4d_chunk_first(31) == 0 && nss::bm4d_chunk_first(32) == 32, "chunks");
}

void test_constant() {
    Video v{12, 30, 34, {}, {}, {}};
    v.noisy.assign(static_cast<std::size_t>(v.t) * v.h * v.w, 0.4f);
    v.out.resize(v.noisy.size());
    run(v, 0.05f);
    double worst = 0;
    for (float x : v.out) worst = std::max(worst, std::fabs(double(x) - 0.4));
    std::printf("constant: worst error %.3g\n", worst);
    expect(worst < 1e-5, "a constant volume passes through");
}

void test_denoises() {
    const float sigma = 15.f / 255.f;
    Video v = moving(12, 48, 64, sigma, 3);
    std::vector<float> basic;
    run(v, sigma, &basic);
    const double m_noisy = mse(v.noisy, v.clean), m_basic = mse(basic, v.clean), m_out = mse(v.out, v.clean);
    std::printf("moving: noisy MSE %.3g, basic %.3g, final %.3g\n", m_noisy, m_basic, m_out);
    expect(m_basic < 0.25 * m_noisy, "the hard stage denoises");
    expect(m_out < m_basic, "the Wiener stage improves on the hard stage");
}

void test_small() {
    for (auto dims : {std::vector<int>{1, 1, 1}, {3, 20, 20}, {4, 4, 4}, {5, 6, 7}, {40, 5, 9}}) {
        Video v = moving(dims[0], dims[1], dims[2], 0.05f, 9);
        run(v, 0.05f);
        bool finite = true;
        for (float x : v.out) finite = finite && std::isfinite(x);
        expect(finite, "small volumes give finite output");
    }
}

std::uint64_t digest(const std::vector<float>& v) {
    std::uint64_t h = 1469598103934665603ull;
    for (float f : v) {
        std::uint32_t b;
        std::memcpy(&b, &f, sizeof(b));
        for (int i = 0; i < 4; ++i) h = (h ^ ((b >> (8 * i)) & 255)) * 1099511628211ull;
    }
    return h;
}

void test_targets() {
    const auto caps = nss::backend_caps();
    const std::int64_t all = caps.compiled_targets & caps.runtime_targets;
    std::uint64_t first = 0;
    bool have = false;
    for (std::int64_t bits = all; bits != 0; bits &= bits - 1) {
        const std::int64_t target = bits & -bits;
        if (target == HWY_SCALAR || target == HWY_EMU128) continue;
        hwy::SetSupportedTargetsForTest(target);
        Video v = moving(10, 37, 45, 10.f / 255.f, 5);
        run(v, 10.f / 255.f);
        const std::uint64_t h = digest(v.out);
        std::printf("target %s: digest %016llx\n", hwy::TargetName(target), static_cast<unsigned long long>(h));
        if (have) expect(h == first, "every SIMD target gives the same bits");
        // NEON (Apple clang) and AVX2 (GCC 15) agree on this value.
        expect(h == 0x0a269158eb34fadfull, "the output matches the golden digest");
        first = h;
        have = true;
    }
    hwy::SetSupportedTargetsForTest(0);
}

}  // namespace

int main(int argc, char** argv) {
    if (!hq_test_target(argc, argv)) return 77;
    test_geometry();
    test_constant();
    test_denoises();
    test_small();
    if (argc == 1) test_targets();
    if (failures) return 1;
    std::printf("test_bm4d: ok\n");
    return 0;
}
