#include "nss/cpu_api.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static int check_case(int m, int n, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const int lda = (m + 7) & ~7;
    const int ldu = lda;
    const int ldvt = n;
    std::vector<float> A(static_cast<std::size_t>(lda * n));
    std::vector<float> U(static_cast<std::size_t>(ldu * n));
    std::vector<float> S(static_cast<std::size_t>(n));
    std::vector<float> Vt(static_cast<std::size_t>(ldvt * n));
    double anorm = 0.0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            const float v = dist(rng);
            A[static_cast<std::size_t>(i + j * lda)] = v;
            anorm += static_cast<double>(v) * v;
        }
    }
    anorm = std::sqrt(anorm);
    if (nss::svd_economy(m, n, A.data(), lda, U.data(), ldu, S.data(), Vt.data(), ldvt) != 0) {
        std::fprintf(stderr, "svd_economy failed m=%d n=%d\n", m, n);
        return 1;
    }
    double rnorm = 0.0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            double rec = 0.0;
            const int kmax = std::min(m, n);
            for (int k = 0; k < kmax; ++k) {
                rec += static_cast<double>(U[static_cast<std::size_t>(i + k * ldu)]) * S[static_cast<std::size_t>(k)] *
                       static_cast<double>(Vt[static_cast<std::size_t>(k + j * ldvt)]);
            }
            const double a = A[static_cast<std::size_t>(i + j * lda)];
            const double d = rec - a;
            rnorm += d * d;
        }
    }
    rnorm = std::sqrt(rnorm);
    const double rel = rnorm / (anorm + 1e-12);
    std::printf("svd m=%d n=%d rel=%.6g\n", m, n, rel);
    if (!(rel < 5e-4)) {
        std::fprintf(stderr, "reconstruction residual too large: %g\n", rel);
        return 1;
    }
    return 0;
}

int main() {
    std::mt19937 rng(12345);
    if (check_case(64, 8, rng) != 0) {
        return 1;
    }
    if (check_case(16, 4, rng) != 0) {
        return 1;
    }
    if (check_case(9, 9, rng) != 0) {
        return 1;
    }
    return 0;
}
