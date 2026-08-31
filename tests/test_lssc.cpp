#include "nss/cpu_api.hpp"
#include "nss/cpu_lssc.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static int fail(const char* msg) {
    std::fprintf(stderr, "%s\n", msg);
    return 1;
}

static int test_cluster_assigns_every_patch() {
    constexpr int m = 16;
    constexpr int n = 20;
    constexpr int lda = 16;
    constexpr int k = nss::kLsscDefaultClusters;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> patches(static_cast<std::size_t>(lda * n), 0.f);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            patches[static_cast<std::size_t>(i + j * lda)] = dist(rng) + 0.25f * static_cast<float>(j % 4);
        }
    }
    std::vector<int> assign(static_cast<std::size_t>(n), -1);
    std::vector<int> counts(static_cast<std::size_t>(k), 0);
    nss::lssc_cluster(patches.data(), m, n, lda, k, assign.data(), counts.data());
    int sum = 0;
    for (int c = 0; c < k; ++c) {
        if (counts[static_cast<std::size_t>(c)] < 0) {
            return fail("cluster negative count");
        }
        sum += counts[static_cast<std::size_t>(c)];
    }
    if (sum != n) {
        return fail("cluster counts do not cover every patch");
    }
    for (int i = 0; i < n; ++i) {
        const int a = assign[static_cast<std::size_t>(i)];
        if (a < 0 || a >= k) {
            return fail("cluster left a patch unassigned");
        }
    }

    constexpr int k3 = 3;
    std::fill(assign.begin(), assign.end(), -1);
    std::vector<int> counts3(k3, 0);
    nss::lssc_cluster(patches.data(), m, n, lda, k3, assign.data(), counts3.data());
    int sum3 = 0;
    for (int c = 0; c < k3; ++c) {
        sum3 += counts3[static_cast<std::size_t>(c)];
    }
    if (sum3 != n) {
        return fail("k=3 cluster missed patches");
    }
    for (int i = 0; i < n; ++i) {
        if (assign[static_cast<std::size_t>(i)] < 0 || assign[static_cast<std::size_t>(i)] >= k3) {
            return fail("k=3 unassigned patch");
        }
    }
    std::printf("cluster assigned n=%d k=%d and k=%d\n", n, k, k3);
    return 0;
}

static int test_reconstruct_finite() {
    constexpr int block = 8;
    constexpr int m = block * block;
    constexpr int n = 12;
    constexpr int atoms = 16;
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(0.1f, 0.9f);
    std::vector<float> patches(static_cast<std::size_t>(m * n));
    for (float& v : patches) {
        v = dist(rng);
    }
    std::vector<float> D(static_cast<std::size_t>(m * atoms), 0.f);
    nss::lssc_dict_init(D.data(), m, atoms, m, patches.data(), n, m, block, 1, 123u);
    for (int a = 0; a < atoms; ++a) {
        for (int i = 0; i < m; ++i) {
            if (!std::isfinite(D[static_cast<std::size_t>(i + a * m)])) {
                return fail("dict produced non-finite atom");
            }
        }
    }
    nss::lssc_reconstruct(patches.data(), m, n, m, D.data(), atoms, m, 3.f / 255.f);
    for (int i = 0; i < m * n; ++i) {
        if (!std::isfinite(patches[static_cast<std::size_t>(i)])) {
            return fail("reconstruct produced non-finite values");
        }
    }

    std::vector<float> A(static_cast<std::size_t>(atoms * n), 0.f);
    for (int i = 0; i < atoms * n; ++i) {
        A[static_cast<std::size_t>(i)] = dist(rng) - 0.5f;
    }
    nss::lssc_group_soft(A.data(), atoms, n, atoms, 0.25f);
    for (int i = 0; i < atoms * n; ++i) {
        if (!std::isfinite(A[static_cast<std::size_t>(i)])) {
            return fail("group_soft non-finite");
        }
    }

    std::vector<float> coef(static_cast<std::size_t>(atoms), 0.f);
    const int ksel = nss::lssc_omp(patches.data(), m, D.data(), atoms, m, 8, coef.data());
    if (ksel < 0) {
        return fail("omp failed");
    }
    for (int i = 0; i < atoms; ++i) {
        if (!std::isfinite(coef[static_cast<std::size_t>(i)])) {
            return fail("omp non-finite");
        }
    }
    std::printf("reconstruct finite atoms=%d omp_sel=%d\n", atoms, ksel);
    return 0;
}

static double mse(const float* a, const float* b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        const double d = static_cast<double>(a[i] - b[i]);
        s += d * d;
    }
    return s / static_cast<double>(n);
}

static int test_denoise_psnr() {
    constexpr int w = 32;
    constexpr int h = 32;
    constexpr int block = 8;
    constexpr int step = 4;
    constexpr float noise = 0.08f;
    std::vector<float> clean(static_cast<std::size_t>(w * h));
    std::vector<float> noisy(static_cast<std::size_t>(w * h));
    std::vector<float> num(static_cast<std::size_t>(w * h), 0.f);
    std::vector<float> den(static_cast<std::size_t>(w * h), 0.f);
    std::vector<float> out(static_cast<std::size_t>(w * h), 0.f);
    std::mt19937 rng(42);
    std::normal_distribution<float> gauss(0.f, noise);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float v = 0.35f + 0.2f * std::sin(static_cast<float>(x) * 0.2f) +
                            0.2f * std::sin(static_cast<float>(y) * 0.15f);
            clean[static_cast<std::size_t>(y * w + x)] = v;
            noisy[static_cast<std::size_t>(y * w + x)] = v + gauss(rng);
        }
    }
    nss::lssc_denoise_plane(noisy.data(), w, h, w, num.data(), den.data(), w, block, step, noise);
    nss::aggregate_finish(out.data(), num.data(), den.data(), noisy.data(), w, h, w, w);
    for (int i = 0; i < w * h; ++i) {
        if (!std::isfinite(out[static_cast<std::size_t>(i)])) {
            return fail("denoise plane non-finite");
        }
    }
    const double mse_n = mse(noisy.data(), clean.data(), w * h);
    const double mse_d = mse(out.data(), clean.data(), w * h);
    const double psnr_n = 10.0 * std::log10(1.0 / (mse_n + 1e-18));
    const double psnr_d = 10.0 * std::log10(1.0 / (mse_d + 1e-18));
    std::printf("lssc psnr_noisy=%.3f psnr_den=%.3f\n", psnr_n, psnr_d);
    if (!(psnr_d > psnr_n + 0.2)) {
        return fail("PSNR vs clean must beat noisy by > 0.2 dB");
    }
    return 0;
}

int main() {
    if (test_cluster_assigns_every_patch() != 0) {
        return 1;
    }
    if (test_reconstruct_finite() != 0) {
        return 1;
    }
    if (test_denoise_psnr() != 0) {
        return 1;
    }
    {
        constexpr int w = 128;
        constexpr int h = 128;
        constexpr int block = 8;
        constexpr int step = 8;
        constexpr float sigma = 25.f / 255.f;
        std::vector<float> clean(static_cast<std::size_t>(w * h));
        std::vector<float> noisy(static_cast<std::size_t>(w * h));
        std::vector<float> num(static_cast<std::size_t>(w * h), 0.f);
        std::vector<float> den(static_cast<std::size_t>(w * h), 0.f);
        std::vector<float> out(static_cast<std::size_t>(w * h), 0.f);
        std::mt19937 rng(1);
        std::normal_distribution<float> gauss(0.f, sigma);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const float v = 0.35f + 0.25f * std::sin(static_cast<float>(x) / 12.f) *
                                            std::cos(static_cast<float>(y) / 9.f);
                clean[static_cast<std::size_t>(y * w + x)] = v;
                noisy[static_cast<std::size_t>(y * w + x)] = v + gauss(rng);
            }
        }
        nss::lssc_denoise_plane(noisy.data(), w, h, w, num.data(), den.data(), w, block, step, sigma);
        nss::aggregate_finish(out.data(), num.data(), den.data(), noisy.data(), w, h, w, w);
        double mn = 1e30;
        double mx = -1e30;
        for (int i = 0; i < w * h; ++i) {
            const float v = out[static_cast<std::size_t>(i)];
            if (!std::isfinite(v) || std::fabs(v) > 8.f) {
                return fail("lssc 128 ISTA exploded");
            }
            mn = std::min(mn, static_cast<double>(v));
            mx = std::max(mx, static_cast<double>(v));
        }
        const double psnr_n = 10.0 * std::log10(1.0 / (mse(noisy.data(), clean.data(), w * h) + 1e-18));
        const double psnr_d = 10.0 * std::log10(1.0 / (mse(out.data(), clean.data(), w * h) + 1e-18));
        std::printf("lssc 128 psnr_noisy=%.3f psnr_den=%.3f range=[%.3f,%.3f]\n", psnr_n, psnr_d, mn, mx);
        if (!(psnr_d > psnr_n + 0.2)) {
            return fail("lssc 128 must beat noisy");
        }
    }
    std::printf("test_lssc ok\n");
    return 0;
}
