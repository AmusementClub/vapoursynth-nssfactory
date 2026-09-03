#include "nss/cpu_api.hpp"
#include "nss/cpu_common.hpp"
#include "nss/cpu_ncsr.hpp"
#include "nss/params.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static int fail(const char* msg) {
    std::fprintf(stderr, "%s\n", msg);
    return 1;
}

static int check_sigma0_identity() {
    constexpr int m = 16;
    constexpr int n = 4;
    constexpr int lda = 16;
    std::mt19937 rng(20260830);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> Y(static_cast<std::size_t>(lda * n));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            Y[static_cast<std::size_t>(i + j * lda)] = dist(rng);
        }
    }
    const std::vector<float> orig = Y;
    const int work_n = nss::ncsr_filter_work_floats(m, n);
    std::vector<float> work(static_cast<std::size_t>(work_n));
    if (nss::ncsr_filter_group(Y.data(), m, n, lda, 0.f, nullptr, work.data(), work_n) != 0) {
        return fail("ncsr_filter_group sigma=0 failed");
    }
    double rnorm = 0.0;
    double anorm = 0.0;
    int nonfinite = 0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            const float v = Y[static_cast<std::size_t>(i + j * lda)];
            if (!std::isfinite(v)) {
                ++nonfinite;
            }
            const double a = orig[static_cast<std::size_t>(i + j * lda)];
            const double d = static_cast<double>(v) - a;
            rnorm += d * d;
            anorm += a * a;
        }
    }
    rnorm = std::sqrt(rnorm);
    anorm = std::sqrt(anorm);
    const double rel = rnorm / (anorm + 1e-12);
    std::printf("ncsr identity m=%d n=%d lda=%d rel=%.6g\n", m, n, lda, rel);
    if (nonfinite != 0) {
        return fail("sigma=0 produced non-finite values");
    }
    if (!(rel < 5e-4)) {
        return fail("sigma=0 should be approximately identity");
    }
    return 0;
}

static int check_two_outer_iters_finite() {
    constexpr int w = 24;
    constexpr int h = 24;
    constexpr int block = 8;
    constexpr int step = 8;
    constexpr int group = 4;
    constexpr int bm_range = 3;
    constexpr float sigma = nss::kNcsrDefaultSigma / 255.f;
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(0.1f, 0.9f);
    std::vector<float> src(static_cast<std::size_t>(w * h));
    std::vector<float> dst(static_cast<std::size_t>(w * h), 0.f);
    for (float& v : src) {
        v = dist(rng);
    }
    const int work_n = nss::ncsr_denoise_work_floats(w, h, block, group);
    std::vector<float> work(static_cast<std::size_t>(work_n));
    nss::ncsr_denoise_plane(src.data(), w, h, w, dst.data(), w, block, step, group, bm_range, sigma,
                            nss::kNcsrDefaultIters, nss::kNcsrDefaultDelta, work.data(), work_n);
    int nonfinite = 0;
    for (int i = 0; i < w * h; ++i) {
        if (!std::isfinite(dst[static_cast<std::size_t>(i)])) {
            ++nonfinite;
        }
    }
    std::printf("ncsr two-iter finite=%d n=%d\n", nonfinite == 0, w * h);
    if (nonfinite != 0) {
        return fail("two outer-iter kernel path produced non-finite values");
    }
    return 0;
}

static int check_neighbor_nlm() {
    constexpr int m = 16;
    constexpr int n = 3;
    constexpr int lda = 16;
    std::vector<float> Y(static_cast<std::size_t>(lda * n), 0.f);
    for (int i = 0; i < m; ++i) {
        Y[static_cast<std::size_t>(i)] = 0.4f + 0.01f * static_cast<float>(i);
        Y[static_cast<std::size_t>(i + lda)] = Y[static_cast<std::size_t>(i)];
        Y[static_cast<std::size_t>(i + 2 * lda)] = 0.9f;
    }
    float dist[3] = {0.f, 0.f, 100.f};
    const int work_n = nss::ncsr_filter_work_floats(m, n);
    std::vector<float> work(static_cast<std::size_t>(work_n));
    if (nss::ncsr_filter_group(Y.data(), m, n, lda, 0.05f, dist, work.data(), work_n) != 0) {
        return fail("ncsr neighbor filter failed");
    }
    double d01 = 0.0;
    double d02 = 0.0;
    for (int i = 0; i < m; ++i) {
        const double a = Y[static_cast<std::size_t>(i)];
        const double b = Y[static_cast<std::size_t>(i + lda)];
        const double c = Y[static_cast<std::size_t>(i + 2 * lda)];
        d01 += (a - b) * (a - b);
        d02 += (a - c) * (a - c);
    }
    std::printf("ncsr neighbor d01=%.6g d02=%.6g\n", d01, d02);
    if (d01 > 1e-6) {
        return fail("identical neighbors must stay together after coding-domain NLM");
    }
    return 0;
}

static int check_scn_tau() {
    constexpr int r = 1;
    constexpr int n = 4;
    float B[4] = {1.f, 1.f, 1.f, 5.f};
    const float orig[4] = {1.f, 1.f, 1.f, 5.f};
    nss::ncsr_centralize_codes(B, r, n, r, 0.5f, nullptr, nullptr);
    // β = 2, residual = {-1,-1,-1,3}, soft 0.5 → {-0.5,-0.5,-0.5,2.5} + 2
    const float expect[4] = {1.5f, 1.5f, 1.5f, 4.5f};
    for (int j = 0; j < n; ++j) {
        if (std::fabs(B[j] - expect[j]) > 1e-5f) {
            return fail("ncsr_centralize_codes scalar tau");
        }
        if (!std::isfinite(orig[j])) {
            return fail("ncsr tau orig");
        }
    }
    std::printf("ncsr scn/centralize tau ok\n");
    return 0;
}

static int check_energy_shrink() {
    constexpr int m = 16;
    constexpr int n = 4;
    constexpr int lda = 24;
    std::vector<float> Y(static_cast<std::size_t>(lda * n), 0.f);
    double e0 = 0.0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            const float v = 0.15f * static_cast<float>((i + 3 * j) % 7);
            Y[static_cast<std::size_t>(i + j * lda)] = v;
            e0 += static_cast<double>(v) * v;
        }
    }
    const int work_n = nss::ncsr_filter_work_floats(m, n);
    std::vector<float> work(static_cast<std::size_t>(work_n));
    if (nss::ncsr_filter_group(Y.data(), m, n, lda, 0.2f, nullptr, work.data(), work_n) != 0) {
        return fail("ncsr_filter_group sigma>0 failed");
    }
    double e1 = 0.0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            const float v = Y[static_cast<std::size_t>(i + j * lda)];
            if (!std::isfinite(v)) {
                return fail("ncsr energy non-finite");
            }
            e1 += static_cast<double>(v) * v;
        }
    }
    std::printf("ncsr energy %.6g -> %.6g\n", e0, e1);
    if (!(e1 < e0)) {
        return fail("ncsr should shrink energy");
    }
    return 0;
}

static int check_group_weights() {
    constexpr int m = 9;
    constexpr int lda = 12;
    constexpr float h = 1.75f;
    std::vector<float> group(static_cast<std::size_t>(lda * nss::kSvdMaxN), 0.f);
    for (int col = 0; col < nss::kSvdMaxN; ++col) {
        for (int row = 0; row < m; ++row) {
            group[static_cast<std::size_t>(row + col * lda)] =
                0.03125f * static_cast<float>((row * 5 + col * 3) % 19);
        }
    }
    const int sizes[] = {1, 3, 7, 8, 15, 16, 31, 32};
    for (int n : sizes) {
        std::vector<float> distance(static_cast<std::size_t>(n));
        std::vector<float> weights(static_cast<std::size_t>(n));
        for (int col = 0; col < n; ++col) {
            distance[static_cast<std::size_t>(col)] = 0.125f * static_cast<float>(col * col + 1);
        }
        const float sum = nss::ncsr_group_weights(distance.data(), nullptr, m, n, lda, h, weights.data());
        double expected_sum = 0.0;
        for (int col = 0; col < n; ++col) {
            const float expected = std::exp(-distance[static_cast<std::size_t>(col)] / h);
            expected_sum += expected;
            if (std::fabs(weights[static_cast<std::size_t>(col)] - expected) > 3e-5f) {
                return fail("ncsr explicit group weight differs from exp oracle");
            }
        }
        if (std::fabs(static_cast<double>(sum) - expected_sum) > 2e-4 * expected_sum + 1e-6) {
            return fail("ncsr explicit group weight sum differs from oracle");
        }

        const float measured_sum =
            nss::ncsr_group_weights(nullptr, group.data(), m, n, lda, h, weights.data());
        expected_sum = 0.0;
        for (int col = 0; col < n; ++col) {
            double ssd = 0.0;
            for (int row = 0; row < m; ++row) {
                const double delta = group[static_cast<std::size_t>(row + col * lda)] -
                                     group[static_cast<std::size_t>(row)];
                ssd += delta * delta;
            }
            const float expected = std::exp(-static_cast<float>(ssd) / h);
            expected_sum += expected;
            if (std::fabs(weights[static_cast<std::size_t>(col)] - expected) > 3e-5f) {
                return fail("ncsr measured group weight differs from exp oracle");
            }
        }
        if (std::fabs(weights[0] - 1.f) > 3e-5f ||
            std::fabs(static_cast<double>(measured_sum) - expected_sum) > 2e-4 * expected_sum + 1e-6) {
            return fail("ncsr measured group weight contract");
        }
    }
    std::printf("ncsr group weights explicit/measured ok\n");
    return 0;
}

int main() {
    if (check_sigma0_identity() != 0) {
        return 1;
    }
    if (check_energy_shrink() != 0) {
        return 1;
    }
    if (check_two_outer_iters_finite() != 0) {
        return 1;
    }
    if (check_neighbor_nlm() != 0) {
        return 1;
    }
    if (check_scn_tau() != 0) {
        return 1;
    }
    if (check_group_weights() != 0) {
        return 1;
    }
    std::printf("test_ncsr ok\n");
    return 0;
}
