// SPDX-License-Identifier: GPL-2.0-only
// DA3D kernel: the 64x64 DFT against a direct DFT, the reference
// exponential, the tile grid, flat and noise-free inputs, a denoising sanity
// check, and the same bits from every SIMD target.
#include "nss/cpu_da3d.hpp"
#include "nss/params/da3d.hpp"
#include "hwy/targets.h"
#include "hq_test_target.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

namespace {

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

struct Buffer {
    std::vector<float> storage;
    float* data;
    explicit Buffer(std::size_t floats) : storage(floats + 16) {
        auto p = reinterpret_cast<std::uintptr_t>(storage.data());
        data = storage.data() + ((64 - p % 64) % 64) / sizeof(float);
    }
};

void test_dft() {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    Buffer re(4096), im(4096), scratch(8192);
    std::vector<float> x_re(4096), x_im(4096);
    for (int i = 0; i < 4096; ++i) {
        x_re[i] = re.data[i] = u(rng);
        x_im[i] = im.data[i] = u(rng);
    }
    nss::da3d_dft64x64(re.data, im.data, false, scratch.data);
    const double pi = 3.14159265358979323846;
    double worst = 0, scale = 0;
    for (int fr = 0; fr < 64; ++fr) {
        for (int fc = 0; fc < 64; fc += 7) {
            double sr = 0, si = 0;
            for (int row = 0; row < 64; ++row)
                for (int col = 0; col < 64; ++col) {
                    const double a = -2 * pi * (fr * row + fc * col) / 64.0;
                    const double xr = x_re[row * 64 + col], xi = x_im[row * 64 + col];
                    sr += xr * std::cos(a) - xi * std::sin(a);
                    si += xr * std::sin(a) + xi * std::cos(a);
                }
            const int o = fc * 64 + fr;  // the transform's layout: [fc][fr]
            worst = std::max(worst, std::hypot(re.data[o] - sr, im.data[o] - si));
            scale = std::max(scale, std::hypot(sr, si));
        }
    }
    expect(worst < 2e-5 * scale, "forward DFT matches the direct DFT");
    nss::da3d_dft64x64(re.data, im.data, true, scratch.data);
    double back = 0;
    for (int i = 0; i < 4096; ++i)
        back = std::max(back, std::hypot(re.data[i] / 4096.0 - x_re[i], im.data[i] / 4096.0 - x_im[i]));
    expect(back < 1e-5, "inverse DFT restores the input");
}

float reference_fastexp(float x) {  // the IPOL code, where it is defined
    int result = static_cast<int>(12102203 * x) + 1065353216;
    result *= result > 0;
    float out;
    std::memcpy(&out, &result, sizeof(out));
    return out;
}

void test_fastexp() {
    for (float x = -170.f; x <= 0.f; x += 0.013f) {
        const float a = nss::da3d_fastexp(x), b = reference_fastexp(x);
        if (std::memcmp(&a, &b, sizeof(a)) != 0) {
            std::fprintf(stderr, "fastexp(%g) = %g, reference %g\n", x, a, b);
            expect(false, "fastexp matches the reference");
            return;
        }
    }
    expect(nss::da3d_fastexp(0.f) == 1.f, "fastexp(0) = 1");
    expect(nss::da3d_fastexp(-INFINITY) == 0.f, "fastexp(-inf) = 0");
    expect(nss::da3d_fastexp(-1e30f) == 0.f, "fastexp(-1e30) = 0");
}

void test_tiles() {
    expect(nss::da3d_tile_count(1080) == 8 && nss::da3d_tile_count(1920) == 15, "1080p grid is 8 x 15");
    expect(nss::da3d_tile_count(1) == 1 && nss::da3d_tile_count(191) == 1 && nss::da3d_tile_count(192) == 2,
           "small grids");
    expect(nss::da3d_tile_max(1080) == 135 && nss::da3d_tile_max(191) == 191, "largest tile");
}

struct Image {
    int w, h, c;
    std::vector<float> noisy, guide, out;
    Image(int w_, int h_, int c_) : w(w_), h(h_), c(c_), noisy(w_ * h_ * c_), guide(w_ * h_ * c_), out(w_ * h_ * c_) {}
    nss::Da3dImage view(float sigma, bool rgb) {
        nss::Da3dImage v;
        v.channels = c;
        v.width = w;
        v.height = h;
        v.rgb = rgb;
        v.sigma = sigma;
        for (int p = 0; p < c; ++p) {
            v.noisy[p] = noisy.data() + p * w * h;
            v.guide[p] = guide.data() + p * w * h;
            v.out[p] = out.data() + p * w * h;
            v.noisy_stride[p] = v.guide_stride[p] = v.out_stride[p] = w;
        }
        return v;
    }
};

nss::Da3dStats run(Image& img, float sigma, bool rgb = false) {
    Buffer work(nss::da3d_work_floats(img.w, img.h, img.c));
    nss::Da3dStats stats;
    nss::da3d_denoise(img.view(sigma, rgb), work.data, nss::da3d_work_floats(img.w, img.h, img.c), &stats);
    return stats;
}

double mse(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0;
    for (std::size_t i = 0; i < a.size(); ++i) s += (double(a[i]) - b[i]) * (double(a[i]) - b[i]);
    return s / a.size();
}

// A constant plane passes through exactly up to rounding; so does a ramp away
// from the borders (the mirrored padding folds it there).
void test_flat_and_ramp() {
    for (int c : {1, 3}) {
        for (bool ramp : {false, true}) {
            Image img(300, 230, c);
            for (int p = 0; p < c; ++p)
                for (int y = 0; y < img.h; ++y)
                    for (int x = 0; x < img.w; ++x) {
                        const float v = 0.25f + 0.1f * p + (ramp ? 0.002f * x - 0.001f * y : 0.f);
                        img.noisy[(p * img.h + y) * img.w + x] = img.guide[(p * img.h + y) * img.w + x] = v;
                    }
            run(img, 5.f / 255.f, c == 3);
            double worst = 0;
            for (int p = 0; p < c; ++p)
                for (int y = 80; y < img.h - 80; ++y)
                    for (int x = 80; x < img.w - 80; ++x) {
                        const std::size_t i = (p * img.h + y) * img.w + x;
                        worst = std::max(worst, std::fabs(double(img.out[i]) - img.noisy[i]));
                    }
            std::printf("%s, channels %d: worst interior error %.3g\n", ramp ? "ramp" : "flat", c, worst);
            expect(worst < 1e-5, ramp ? "a noise-free ramp passes through" : "a constant plane passes through");
        }
    }
}

// Platform-independent inputs (no libm, no std::normal_distribution): the
// digests printed by test_targets can be compared across hosts.
struct Gauss {
    std::mt19937 rng;
    explicit Gauss(std::uint32_t seed) : rng(seed) {}
    float operator()() {  // Irwin-Hall: the sum of 12 uniforms minus 6
        float s = 0.f;
        for (int i = 0; i < 12; ++i) s += static_cast<float>(rng() >> 8) * (1.f / 16777216.f);
        return s - 6.f;
    }
};

Image noisy_scene(int w, int h, int c, float sigma, std::uint32_t seed, std::vector<float>* clean_out) {
    Image img(w, h, c);
    Gauss n(seed);
    std::vector<float> clean(w * h * c);
    for (int p = 0; p < c; ++p)
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const int t = (x * (p + 2) + y * 3) % 64;  // a triangle wave
                float v = 0.3f + 0.05f * p + 0.3f * ((x / 23 + y / 17) % 2) + 0.004f * static_cast<float>(t < 32 ? t : 64 - t);
                if ((x - 60) * (x - 60) + (y - 40) * (y - 40) < 400) v = 0.9f - 0.1f * p;
                clean[(p * h + y) * w + x] = v;
            }
    for (std::size_t i = 0; i < clean.size(); ++i) {
        img.noisy[i] = clean[i] + sigma * n();
        img.guide[i] = clean[i] + 0.4f * sigma * n();  // a decent, imperfect guide
    }
    *clean_out = clean;
    return img;
}

void test_denoises() {
    for (int c : {1, 3}) {
        std::vector<float> clean;
        const float sigma = 20.f / 255.f;
        Image img = noisy_scene(173, 131, c, sigma, 11 + c, &clean);
        const auto stats = run(img, sigma, c == 3);
        const double m_guide = mse(img.guide, clean), m_out = mse(img.out, clean);
        std::printf("channels %d: guide MSE %.3g, DA3D %.3g, patches %lld (guide only %lld)\n", c, m_guide, m_out,
                    static_cast<long long>(stats.patches), static_cast<long long>(stats.guide_only));
        expect(m_out < m_guide, c == 1 ? "DA3D improves on the guide (gray)" : "DA3D improves on the guide (RGB)");
        bool finite = true;
        for (float v : img.out) finite = finite && std::isfinite(v);
        expect(finite, "output is finite");
    }
}

void test_tiny() {
    for (auto [w, h] : {std::pair{1, 1}, std::pair{5, 3}, std::pair{2, 70}}) {
        std::vector<float> clean;
        Image img = noisy_scene(w, h, 1, 0.05f, 3, &clean);
        run(img, 0.05f);
        bool finite = true;
        for (float v : img.out) finite = finite && std::isfinite(v);
        expect(finite, "tiny planes give finite output");
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

// Every SIMD target (not the scalar fallback, whose MulAdd is not fused)
// gives the same bits.
void test_targets() {
    const auto caps = nss::backend_caps();
    const std::int64_t all = caps.compiled_targets & caps.runtime_targets;
    std::uint64_t first = 0;
    bool have = false;
    for (std::int64_t bits = all; bits != 0; bits &= bits - 1) {
        const std::int64_t target = bits & -bits;
        if (target == HWY_SCALAR || target == HWY_EMU128) continue;
        hwy::SetSupportedTargetsForTest(target);
        std::vector<float> clean;
        Image img = noisy_scene(140, 150, 3, 12.f / 255.f, 5, &clean);
        run(img, 12.f / 255.f, true);
        Image gray = noisy_scene(140, 150, 1, 12.f / 255.f, 6, &clean);
        run(gray, 12.f / 255.f);
        std::vector<float> both = img.out;
        both.insert(both.end(), gray.out.begin(), gray.out.end());
        const std::uint64_t h = digest(both);
        std::printf("target %s: digest %016llx\n", hwy::TargetName(target), static_cast<unsigned long long>(h));
        if (have) expect(h == first, "every SIMD target gives the same bits");
        // NEON (Apple clang) and AVX2 (GCC 15) agree on this value.
        expect(h == 0xe64809064d4c5fe8ull, "the output matches the golden digest");
        first = h;
        have = true;
    }
    hwy::SetSupportedTargetsForTest(0);
}

}  // namespace

int main(int argc, char** argv) {
    if (!hq_test_target(argc, argv)) return 77;
    test_dft();
    test_fastexp();
    test_tiles();
    test_flat_and_ramp();
    test_denoises();
    test_tiny();
    if (argc == 1) test_targets();
    if (failures) return 1;
    std::printf("test_da3d: ok\n");
    return 0;
}
