// SPDX-License-Identifier: GPL-2.0-only
// PNLM kernel: the exponential, the reference grid, the kernel against a
// direct double-precision implementation of the model (exhaustive search,
// sorted matches, weights, flat rule, tent aggregation), flat and noise-free
// planes, the temporal window, a denoising sanity check, and the same bits
// from every SIMD target.
#include "nss/backend.hpp"
#include "nss/cpu_pnlm.hpp"
#include "nss/params/pnlm.hpp"
#include "hwy/targets.h"
#include "hq_test_target.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <tuple>
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

// Gaussian noise the same on every platform (std::normal_distribution is
// not): Irwin-Hall, the sum of 12 uniforms minus 6.
struct Gauss {
    std::mt19937 rng;
    explicit Gauss(std::uint32_t seed) : rng(seed) {}
    float operator()() {
        float s = 0.f;
        for (int i = 0; i < 12; ++i) s += static_cast<float>(rng() >> 8) * (1.f / 16777216.f);
        return s - 6.f;
    }
};

// A clip of `frames` planes w x h: steps, a ramp and a disc, moving 1 px a
// frame, with Gaussian noise of sigma (8-bit units).
struct Clip {
    int w, h, frames;
    std::vector<std::vector<float>> noisy, clean;
};
Clip scene(int w, int h, int frames, float sigma, unsigned seed) {
    Clip c{w, h, frames, {}, {}};
    Gauss g(seed);
    for (int t = 0; t < frames; ++t) {
        std::vector<float> cl(static_cast<std::size_t>(w) * h), no(cl.size());
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const int xs = x + t;
                float v = 0.25f + 0.4f * (((xs / 9) + (y / 7)) % 2) + 0.002f * xs;
                if ((xs - w / 2) * (xs - w / 2) + (y - h / 2) * (y - h / 2) < h * h / 9) v = 0.8f;
                cl[y * w + x] = v;
                no[y * w + x] = v + sigma / 255.f * g();
            }
        c.clean.push_back(cl);
        c.noisy.push_back(no);
    }
    return c;
}

std::vector<float> run(const Clip& c, int center, int frames_lo, int frames_hi, float sigma, int a, int block, int group) {
    std::vector<const float*> planes;
    std::vector<std::ptrdiff_t> strides;
    for (int t = frames_lo; t <= frames_hi; ++t) {
        planes.push_back(c.noisy[t].data());
        strides.push_back(c.w);
    }
    std::vector<float> out(static_cast<std::size_t>(c.w) * c.h);
    nss::PnlmImage img;
    img.width = c.w;
    img.height = c.h;
    img.frames = frames_hi - frames_lo + 1;
    img.center = center - frames_lo;
    img.src = planes.data();
    img.src_stride = strides.data();
    img.out = out.data();
    img.out_stride = c.w;
    img.sigma = sigma;
    img.a = a;
    img.block = block;
    img.group = group;
    const std::size_t need = nss::pnlm_work_floats(c.w, block);
    Buffer work(need);
    nss::pnlm_denoise(img, work.data, need);
    return out;
}

// The model written out directly, in double precision.
std::vector<double> oracle(const Clip& c, int center, int lo, int hi, float sigma8, int a, int p, int group) {
    const int w = c.w, h = c.h, area = p * p;
    const double s = sigma8 / 255.0, h2 = (2 * s) * (2 * s);
    std::vector<int> xs, ys;
    for (int i = 0; i < nss::pnlm_grid_count(w, p); ++i) xs.push_back(nss::pnlm_grid_pos(i, w, p));
    for (int i = 0; i < nss::pnlm_grid_count(h, p); ++i) ys.push_back(nss::pnlm_grid_pos(i, h, p));
    std::vector<double> num(static_cast<std::size_t>(w) * h), den(num.size());
    auto px = [&](int t, int y, int x) { return static_cast<double>(c.noisy[t][y * w + x]); };
    for (int y0 : ys)
        for (int x0 : xs) {
            std::vector<std::tuple<double, int, int, int>> cand;
            for (int t = lo; t <= hi; ++t)
                for (int y = std::max(0, y0 - a); y <= std::min(h - p, y0 + a); ++y)
                    for (int x = std::max(0, x0 - a); x <= std::min(w - p, x0 + a); ++x) {
                        if (t == center && x == x0 && y == y0) continue;
                        double d = 0;
                        for (int r = 0; r < p; ++r)
                            for (int q = 0; q < p; ++q) {
                                const double e = px(center, y0 + r, x0 + q) - px(t, y + r, x + q);
                                d += e * e;
                            }
                        cand.emplace_back(d, t - lo, y, x);
                    }
            std::sort(cand.begin(), cand.end());
            cand.resize(std::min<std::size_t>(cand.size(), static_cast<std::size_t>(group - 1)));
            cand.insert(cand.begin(), std::make_tuple(0.0, center - lo, y0, x0));
            std::vector<double> acc(area);
            double wsum = 0, s1 = 0, s2 = 0;
            for (std::size_t i = 0; i < cand.size(); ++i) {
                const auto [d, t, y, x] = cand[i];
                const double wt = i == 0 ? 1.0 : std::exp(-std::max(d / area - 2 * s * s, 0.0) / h2);
                wsum += wt;
                for (int r = 0; r < p; ++r)
                    for (int q = 0; q < p; ++q) {
                        const double v = px(t + lo, y + r, x + q);
                        acc[r * p + q] += wt * v;
                        s1 += v;
                        s2 += v * v;
                    }
            }
            const double count = static_cast<double>(cand.size()) * area, mean = s1 / count;
            const double r = std::max(s2 / count - mean * mean, 0.0) / (s * s);
            const double alpha = std::clamp((nss::kPnlmFlatHi - r) / (nss::kPnlmFlatHi - nss::kPnlmFlatLo), 0.0, 1.0);
            for (int r2 = 0; r2 < p; ++r2)
                for (int q = 0; q < p; ++q) {
                    const double e = alpha * mean + (1 - alpha) * acc[r2 * p + q] / wsum;
                    const double tw = (1 - std::abs(2 * r2 + 1 - p) / static_cast<double>(p)) *
                                      (1 - std::abs(2 * q + 1 - p) / static_cast<double>(p));
                    num[(y0 + r2) * w + x0 + q] += tw * e;
                    den[(y0 + r2) * w + x0 + q] += tw;
                }
        }
    for (std::size_t i = 0; i < num.size(); ++i) num[i] /= den[i];
    return num;
}

double psnr(const std::vector<float>& x, const std::vector<float>& y) {
    double e = 0;
    for (std::size_t i = 0; i < x.size(); ++i) e += (static_cast<double>(x[i]) - y[i]) * (static_cast<double>(x[i]) - y[i]);
    return 10 * std::log10(x.size() / e);
}

std::uint64_t digest(const std::vector<float>& v) {
    std::uint64_t h = 1469598103934665603ull;
    for (float f : v) {
        std::uint32_t b;
        std::memcpy(&b, &f, sizeof(b));
        for (int i = 0; i < 4; ++i) h = (h ^ ((b >> (8 * i)) & 0xffu)) * 1099511628211ull;
    }
    return h;
}

void test_exp() {
    double worst = 0;
    for (int i = 0; i < 870000; ++i) {
        const float x = -i * 1e-4f;
        const double ref = std::exp(static_cast<double>(x));
        worst = std::max(worst, std::abs(nss::pnlm_exp(x) - ref) / ref);
    }
    // Rounding x log2 e to float bounds it: about 87 x 1.44 x 2^-24 x ln 2.
    expect(worst < 8e-6, "pnlm_exp within 8e-6 relative on (-87, 0]");
    expect(nss::pnlm_exp(0.f) == 1.f && nss::pnlm_exp(-0.f) == 1.f, "pnlm_exp(0) is 1");
    expect(nss::pnlm_exp(-87.5f) == 0.f && nss::pnlm_exp(-INFINITY) == 0.f && nss::pnlm_exp(NAN) == 0.f,
           "pnlm_exp is 0 below -87, at -inf and for NaN");
    expect(nss::pnlm_exp(-86.9f) >= 1.17549435e-38f, "pnlm_exp stays normal above -87");
}

void test_grid() {
    for (int p : {4, 6, 8})
        for (int n : {p, p + 1, p + p / 2, 37, 64, 1001}) {
            const int count = nss::pnlm_grid_count(n, p);
            bool ok = count >= 1 && nss::pnlm_grid_pos(count - 1, n, p) == n - p && nss::pnlm_grid_pos(0, n, p) == 0;
            std::vector<int> cover(n);
            for (int i = 0; i < count; ++i) {
                if (i && nss::pnlm_grid_pos(i, n, p) <= nss::pnlm_grid_pos(i - 1, n, p)) ok = false;
                for (int k = 0; k < p; ++k) ++cover[nss::pnlm_grid_pos(i, n, p) + k];
            }
            for (int v : cover) ok = ok && v >= 1;
            expect(ok, "grid: increasing, ends at n - p, covers every pixel");
        }
    expect(nss::pnlm_grid_count(3, 4) == 0, "no grid on a plane smaller than the patch");
}

void test_oracle() {
    const Clip c = scene(43, 37, 3, 8.f, 3);
    for (int p : {4, 6, 8})
        for (auto [a, group, lo, hi] : {std::tuple(2, 16, 1, 1), std::tuple(5, 8, 1, 1), std::tuple(3, 32, 0, 2),
                                        std::tuple(1, 5, 0, 1)}) {
            const std::vector<float> got = run(c, 1, lo, hi, 8.f, a, p, group);
            const std::vector<double> want = oracle(c, 1, lo, hi, 8.f, a, p, group);
            double worst = 0;
            for (std::size_t i = 0; i < got.size(); ++i) worst = std::max(worst, std::abs(got[i] - want[i]));
            char what[96];
            std::snprintf(what, sizeof(what), "kernel = model (p %d, a %d, group %d, frames %d): %.2e", p, a, group,
                          hi - lo + 1, worst);
            expect(worst < 2e-5, what);
        }
}

void test_flat() {
    Clip c = scene(40, 30, 1, 0.f, 1);
    for (float& v : c.noisy[0]) v = 0.375f;
    const std::vector<float> out = run(c, 0, 0, 0, 5.f, 2, 6, 16);
    float worst = 0;
    for (float v : out) worst = std::max(worst, std::abs(v - 0.375f));
    expect(worst < 1e-6f, "a constant plane stays constant");
}

void test_denoises() {
    for (int p : {4, 6, 8}) {
        const Clip c = scene(96, 80, 3, 10.f, 9);
        const double before = psnr(c.noisy[1], c.clean[1]);
        const double spatial = psnr(run(c, 1, 1, 1, 10.f, 2, p, 16), c.clean[1]);
        const double temporal = psnr(run(c, 1, 0, 2, 10.f, 3, p, 16), c.clean[1]);
        expect(spatial > before + 2, "denoises (spatial)");
        expect(temporal > spatial, "the neighbouring frames help on a moving scene");
    }
}

void test_temporal_window() {
    // The neighbours of a frame matter only through the frames given.
    Clip c = scene(48, 40, 3, 6.f, 4);
    const std::vector<float> one = run(c, 1, 1, 1, 6.f, 2, 6, 16);
    Clip other = c;
    for (float& v : other.noisy[0]) v = 1 - v;
    for (float& v : other.noisy[2]) v = 1 - v;
    expect(run(other, 1, 1, 1, 6.f, 2, 6, 16) == one, "d = 0 reads only the frame itself");
    expect(run(other, 1, 0, 2, 6.f, 2, 6, 16) != run(c, 1, 0, 2, 6.f, 2, 6, 16), "d = 1 reads its neighbours");
}

std::uint64_t target_digest() {
    const Clip c = scene(101, 67, 3, 7.f, 21);
    std::vector<float> all;
    for (int p : {4, 6, 8}) {
        const std::vector<float> s = run(c, 1, 1, 1, 7.f, 2, p, 16);
        const std::vector<float> t = run(c, 1, 0, 2, 7.f, 5, p, 24);
        all.insert(all.end(), s.begin(), s.end());
        all.insert(all.end(), t.begin(), t.end());
    }
    return digest(all);
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
        const std::uint64_t h = target_digest();
        std::printf("target %s: digest %016llx\n", hwy::TargetName(target), static_cast<unsigned long long>(h));
        if (have) expect(h == first, "every SIMD target gives the same bits");
        // NEON (Apple clang) and AVX2 (GCC 15) agree on this value; nss_cuda gives the same bits.
        expect(h == 0xba2711eb6470ebdeull, "the output matches the golden digest");
        first = h;
        have = true;
    }
    hwy::SetSupportedTargetsForTest(0);
}

}  // namespace

int main(int argc, char** argv) {
    if (!hq_test_target(argc, argv)) return 77;
    test_exp();
    test_grid();
    test_oracle();
    test_flat();
    test_denoises();
    test_temporal_window();
    if (argc == 1) test_targets();
    if (failures) return 1;
    std::printf("test_pnlm: ok\n");
    return 0;
}
