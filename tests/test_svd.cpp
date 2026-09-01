#include "nss/cpu_api.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {

struct SvdCase {
    int m;
    int n;
    const char* name;
    std::function<float(int, int)> value;
};

std::vector<double> reference_singular_values(const std::vector<float>& a, int m, int n, int lda) {
    const int r = std::min(m, n);
    std::vector<double> gram(static_cast<std::size_t>(r * r), 0.0);
    if (m >= n) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k <= j; ++k) {
                double sum = 0.0;
                for (int i = 0; i < m; ++i) {
                    sum += static_cast<double>(a[static_cast<std::size_t>(i + j * lda)]) *
                           static_cast<double>(a[static_cast<std::size_t>(i + k * lda)]);
                }
                gram[static_cast<std::size_t>(k + j * r)] = sum;
                gram[static_cast<std::size_t>(j + k * r)] = sum;
            }
        }
    } else {
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k <= j; ++k) {
                double sum = 0.0;
                for (int col = 0; col < n; ++col) {
                    sum += static_cast<double>(a[static_cast<std::size_t>(j + col * lda)]) *
                           static_cast<double>(a[static_cast<std::size_t>(k + col * lda)]);
                }
                gram[static_cast<std::size_t>(k + j * r)] = sum;
                gram[static_cast<std::size_t>(j + k * r)] = sum;
            }
        }
    }

    // A double-precision symmetric Jacobi oracle is only used by tests.
    for (int sweep = 0; sweep < 128; ++sweep) {
        double max_off = 0.0;
        for (int p = 0; p < r - 1; ++p) {
            for (int q = p + 1; q < r; ++q) {
                const double apq = gram[static_cast<std::size_t>(p + q * r)];
                max_off = std::max(max_off, std::fabs(apq));
                if (std::fabs(apq) <= 1e-15) {
                    continue;
                }
                const double app = gram[static_cast<std::size_t>(p + p * r)];
                const double aqq = gram[static_cast<std::size_t>(q + q * r)];
                const double theta = 0.5 * std::atan2(2.0 * apq, aqq - app);
                const double c = std::cos(theta);
                const double s = std::sin(theta);
                for (int i = 0; i < r; ++i) {
                    const double gip = gram[static_cast<std::size_t>(i + p * r)];
                    const double giq = gram[static_cast<std::size_t>(i + q * r)];
                    gram[static_cast<std::size_t>(i + p * r)] = c * gip - s * giq;
                    gram[static_cast<std::size_t>(i + q * r)] = s * gip + c * giq;
                }
                for (int i = 0; i < r; ++i) {
                    const double gpi = gram[static_cast<std::size_t>(p + i * r)];
                    const double gqi = gram[static_cast<std::size_t>(q + i * r)];
                    gram[static_cast<std::size_t>(p + i * r)] = c * gpi - s * gqi;
                    gram[static_cast<std::size_t>(q + i * r)] = s * gpi + c * gqi;
                }
            }
        }
        if (max_off < 1e-14) {
            break;
        }
    }
    std::vector<double> out(static_cast<std::size_t>(r));
    for (int i = 0; i < r; ++i) {
        out[static_cast<std::size_t>(i)] = std::sqrt(std::max(0.0, gram[static_cast<std::size_t>(i + i * r)]));
    }
    std::sort(out.begin(), out.end(), std::greater<double>());
    return out;
}

double reconstruction_error(const std::vector<float>& a, int m, int n, int lda, const std::vector<float>& u, int ldu,
                            const std::vector<float>& s, const std::vector<float>& vt, int ldvt) {
    const int r = std::min(m, n);
    double err2 = 0.0;
    double norm2 = 0.0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            double rec = 0.0;
            for (int k = 0; k < r; ++k) {
                rec += static_cast<double>(u[static_cast<std::size_t>(i + k * ldu)]) *
                       static_cast<double>(s[static_cast<std::size_t>(k)]) *
                       static_cast<double>(vt[static_cast<std::size_t>(k + j * ldvt)]);
            }
            const double av = a[static_cast<std::size_t>(i + j * lda)];
            const double d = rec - av;
            err2 += d * d;
            norm2 += av * av;
        }
    }
    return std::sqrt(err2) / (std::sqrt(norm2) + 1e-30);
}

double orthogonality_error_u(const std::vector<float>& u, int m, int r, int ldu, const std::vector<float>& s) {
    double worst = 0.0;
    const double scale = std::max(1.0, static_cast<double>(s.empty() ? 0.f : s[0]));
    for (int p = 0; p < r; ++p) {
        for (int q = 0; q < r; ++q) {
            if (s[static_cast<std::size_t>(p)] <= scale * 1e-7 || s[static_cast<std::size_t>(q)] <= scale * 1e-7) {
                continue;
            }
            double dot = 0.0;
            for (int i = 0; i < m; ++i) {
                dot += static_cast<double>(u[static_cast<std::size_t>(i + p * ldu)]) *
                       static_cast<double>(u[static_cast<std::size_t>(i + q * ldu)]);
            }
            const double expected = p == q ? 1.0 : 0.0;
            worst = std::max(worst, std::fabs(dot - expected));
        }
    }
    return worst;
}

double orthogonality_error_v(const std::vector<float>& vt, int n, int r, int ldvt, const std::vector<float>& s) {
    double worst = 0.0;
    const double scale = std::max(1.0, static_cast<double>(s.empty() ? 0.f : s[0]));
    for (int p = 0; p < r; ++p) {
        for (int q = 0; q < r; ++q) {
            if (s[static_cast<std::size_t>(p)] <= scale * 1e-7 || s[static_cast<std::size_t>(q)] <= scale * 1e-7) {
                continue;
            }
            double dot = 0.0;
            for (int j = 0; j < n; ++j) {
                dot += static_cast<double>(vt[static_cast<std::size_t>(p + j * ldvt)]) *
                       static_cast<double>(vt[static_cast<std::size_t>(q + j * ldvt)]);
            }
            const double expected = p == q ? 1.0 : 0.0;
            worst = std::max(worst, std::fabs(dot - expected));
        }
    }
    return worst;
}

int check_case(const SvdCase& tc, double residual_limit, double singular_limit) {
    const int lda = (tc.m + 15) & ~15;
    const int ldu = (tc.m + 7) & ~7;
    const int ldvt = tc.n + 3;
    std::vector<float> a(static_cast<std::size_t>(lda * tc.n), 0.f);
    for (int j = 0; j < tc.n; ++j) {
        for (int i = 0; i < tc.m; ++i) {
            a[static_cast<std::size_t>(i + j * lda)] = tc.value(i, j);
        }
    }
    std::vector<float> u(static_cast<std::size_t>(ldu * tc.n), 0.f);
    std::vector<float> s(static_cast<std::size_t>(std::min(tc.m, tc.n)), 0.f);
    std::vector<float> vt(static_cast<std::size_t>(ldvt * tc.n), 0.f);
    if (nss::svd_economy(tc.m, tc.n, a.data(), lda, u.data(), ldu, s.data(), vt.data(), ldvt) != 0) {
        std::fprintf(stderr, "svd_economy failed for %s (%d x %d)\n", tc.name, tc.m, tc.n);
        return 1;
    }

    const auto ref = reference_singular_values(a, tc.m, tc.n, lda);
    double max_s_err = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double denom = std::max(1.0, ref[i]);
        max_s_err = std::max(max_s_err, std::fabs(static_cast<double>(s[i]) - ref[i]) / denom);
        if (!std::isfinite(s[i]) || (i > 0 && s[i] > s[i - 1] + 1e-6f)) {
            std::fprintf(stderr, "singular values are not finite/ordered for %s\n", tc.name);
            return 1;
        }
    }
    const double rec = reconstruction_error(a, tc.m, tc.n, lda, u, ldu, s, vt, ldvt);
    const double uerr = orthogonality_error_u(u, tc.m, static_cast<int>(ref.size()), ldu, s);
    const double verr = orthogonality_error_v(vt, tc.n, static_cast<int>(ref.size()), ldvt, s);
    std::printf("svd %-18s %2d x %-2d residual=%.6g U=%.6g V=%.6g S=%.6g\n", tc.name, tc.m, tc.n, rec, uerr, verr,
                max_s_err);
    if (!(rec < residual_limit && uerr < 5e-4 && verr < 5e-4 && max_s_err < singular_limit)) {
        std::fprintf(stderr, "SVD quality check failed for %s\n", tc.name);
        return 1;
    }
    return 0;
}

}  // namespace

int main() {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> random(-1.0f, 1.0f);
    int failed = 0;

    failed |= check_case({64, 8, "random-tall", [&](int, int) { return random(rng); }}, 5e-4, 5e-4);
    failed |= check_case({16, 4, "random-small", [&](int, int) { return random(rng); }}, 5e-4, 5e-4);
    failed |= check_case({9, 9, "random-square", [&](int, int) { return random(rng); }}, 5e-4, 5e-4);
    failed |= check_case({4, 7, "random-wide", [&](int, int) { return random(rng); }}, 5e-4, 5e-4);

    failed |= check_case({8, 8, "rank-deficient", [](int i, int j) {
                              const float x = static_cast<float>(i + 1);
                              const float y = static_cast<float>((j % 3) + 1);
                              return (j & 1) ? x * y : x * (j + 1.f);
                          }},
                         5e-4, 5e-4);
    failed |= check_case({8, 8, "ill-conditioned", [](int i, int j) {
                              if (i != j) {
                                  return 0.f;
                              }
                              return std::pow(10.f, -static_cast<float>(j));
                          }},
                         5e-4, 5e-4);
    // Identity plus E(0,2) exercises the pairing round that used to stop early.
    failed |= check_case({8, 8, "identity-plus-02", [](int i, int j) {
                              return (i == j ? 1.f : 0.f) + (i == 0 && j == 2 ? 1.f : 0.f);
                          }},
                         5e-4, 5e-5);
    return failed ? 1 : 0;
}
