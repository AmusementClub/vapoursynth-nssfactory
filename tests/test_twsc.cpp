#include "nss/cpu_twsc.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

static int fail(const char* msg) {
    std::fprintf(stderr, "%s\n", msg);
    return 1;
}

static double frobenius(const float* a, int m, int n, int lda) {
    double s = 0.0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            const double v = a[i + j * lda];
            s += v * v;
        }
    }
    return s;
}

static int check_group(int m, int n, int lda, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> Y(static_cast<std::size_t>(lda * n));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            Y[static_cast<std::size_t>(i + j * lda)] = dist(rng);
        }
    }
    const std::vector<float> orig = Y;
    const int work_n = nss::twsc_pca_soft_work_floats(m, n);
    std::vector<float> work(static_cast<std::size_t>(work_n));

    if (nss::twsc_pca_soft(Y.data(), m, n, lda, 0.f, work.data(), work_n) != 0) {
        return fail("twsc_pca_soft sigma=0 failed");
    }
    double rnorm = 0.0;
    double anorm = 0.0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            const double a = orig[static_cast<std::size_t>(i + j * lda)];
            const double d = static_cast<double>(Y[static_cast<std::size_t>(i + j * lda)]) - a;
            rnorm += d * d;
            anorm += a * a;
        }
    }
    rnorm = std::sqrt(rnorm);
    anorm = std::sqrt(anorm);
    const double rel = rnorm / (anorm + 1e-12);
    std::printf("twsc identity m=%d n=%d lda=%d rel=%.6g\n", m, n, lda, rel);
    if (!(rel < 5e-4)) {
        return fail("sigma=0 should be approximately identity");
    }

    Y = orig;
    if (nss::twsc_pca_soft(Y.data(), m, n, lda, 0.35f, work.data(), work_n) != 0) {
        return fail("twsc_pca_soft sigma>0 failed");
    }
    const double e0 = frobenius(orig.data(), m, n, lda);
    const double e1 = frobenius(Y.data(), m, n, lda);
    std::printf("twsc energy m=%d n=%d lda=%d e0=%.6g e1=%.6g\n", m, n, lda, e0, e1);
    if (!(e1 < e0)) {
        return fail("sigma>0 should shrink energy");
    }
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            if (!std::isfinite(Y[static_cast<std::size_t>(i + j * lda)])) {
                return fail("twsc non-finite");
            }
        }
    }
    return 0;
}

int main() {
    std::mt19937 rng(20260830);
    if (check_group(16, 4, 16, rng) != 0) {
        return 1;
    }
    if (check_group(64, 8, 80, rng) != 0) {
        return 1;
    }
    {
        std::vector<float> a(64 * 8);
        std::vector<float> b = a;
        std::mt19937 rng2(7);
        std::uniform_real_distribution<float> dist2(-1.f, 1.f);
        for (float& v : a) {
            v = dist2(rng2);
        }
        b = a;
        const int wn = nss::twsc_pca_soft_work_floats(64, 8);
        std::vector<float> work(static_cast<std::size_t>(wn));
        float sig_lo[8];
        float sig_hi[8];
        float wlo[8];
        float whi[8];
        for (int i = 0; i < 8; ++i) {
            sig_lo[i] = 0.05f;
            sig_hi[i] = 0.4f;
        }
        if (nss::twsc_pca_soft(a.data(), 64, 8, 64, 0.05f, work.data(), wn, sig_lo, wlo) != 0) {
            return fail("twsc col_sigma lo");
        }
        if (nss::twsc_pca_soft(b.data(), 64, 8, 64, 0.05f, work.data(), wn, sig_hi, whi) != 0) {
            return fail("twsc col_sigma hi");
        }
        double d = 0.0;
        for (int i = 0; i < 64 * 8; ++i) {
            d += std::fabs(static_cast<double>(a[static_cast<std::size_t>(i)] - b[static_cast<std::size_t>(i)]));
        }
        std::printf("twsc lambda/col_sigma L1=%.6g wlo=%.6g whi=%.6g\n", d, static_cast<double>(wlo[0]),
                    static_cast<double>(whi[0]));
        if (!(d > 1e-4)) {
            return fail("different col_sigma must change the estimate");
        }
        if (!(whi[0] < wlo[0])) {
            return fail("W2 = 1/sigma_col should drop as sigma_col grows");
        }
    }
    {
        std::vector<float> a(32 * 4, 0.f);
        std::mt19937 rng3(3);
        std::uniform_real_distribution<float> dist3(-1.f, 1.f);
        for (float& v : a) {
            v = dist3(rng3);
        }
        std::vector<float> b = a;
        const int wn = nss::twsc_pca_soft_work_floats(32, 4);
        std::vector<float> work(static_cast<std::size_t>(wn));
        float row_w[32];
        for (int i = 0; i < 32; ++i) {
            row_w[i] = (i < 16) ? 1.f : 0.25f;
        }
        if (nss::twsc_pca_soft(a.data(), 32, 4, 32, 0.2f, work.data(), wn) != 0) {
            return fail("twsc no W1");
        }
        if (nss::twsc_pca_soft(b.data(), 32, 4, 32, 0.2f, work.data(), wn, nullptr, nullptr, row_w) != 0) {
            return fail("twsc W1");
        }
        double d = 0.0;
        for (int i = 0; i < 32 * 4; ++i) {
            d += std::fabs(static_cast<double>(a[static_cast<std::size_t>(i)] - b[static_cast<std::size_t>(i)]));
        }
        std::printf("twsc W1 L1=%.6g\n", d);
        if (!(d > 1e-4)) {
            return fail("row_w W1 must change the estimate");
        }
    }
    {
        constexpr int m = 16;
        constexpr int n = 4;
        std::vector<float> input(m * n);
        std::mt19937 rng4(11);
        std::uniform_real_distribution<float> dist4(-1.f, 1.f);
        for (float& v : input) {
            v = dist4(rng4);
        }
        const int wn = nss::twsc_pca_soft_work_floats(m, n);
        std::vector<float> work(static_cast<std::size_t>(wn));
        const float bad_values[] = {
            std::numeric_limits<float>::quiet_NaN(),
            -std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
        };
        for (float bad : bad_values) {
            std::vector<float> got = input;
            std::vector<float> want = input;
            if (nss::twsc_pca_soft(got.data(), m, n, m, bad, work.data(), wn) != 0 ||
                nss::twsc_pca_soft(want.data(), m, n, m, 0.f, work.data(), wn) != 0) {
                return fail("twsc non-finite sigma call failed");
            }
            if (got != want) {
                return fail("twsc non-finite sigma was not clamped to zero");
            }
        }

        float bad_sigma[n] = {bad_values[0], bad_values[1], bad_values[2], bad_values[3]};
        float zero_sigma[n]{};
        float got_w[n]{};
        float want_w[n]{};
        std::vector<float> got = input;
        std::vector<float> want = input;
        if (nss::twsc_pca_soft(got.data(), m, n, m, 0.f, work.data(), wn, bad_sigma, got_w) != 0 ||
            nss::twsc_pca_soft(want.data(), m, n, m, 0.f, work.data(), wn, zero_sigma, want_w) != 0) {
            return fail("twsc non-finite col_sigma call failed");
        }
        if (got != want || !std::equal(std::begin(got_w), std::end(got_w), std::begin(want_w))) {
            return fail("twsc non-finite col_sigma was not clamped to zero");
        }
    }
    std::printf("test_twsc ok\n");
    return 0;
}
