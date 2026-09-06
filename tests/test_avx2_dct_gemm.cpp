#include "cpu/bm/dct16.hpp"
#include "cpu/wnnm/jacobi8.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {
constexpr float sentinel = 12345.0f;

bool test_gemm() {
    std::mt19937 rng(1907);
    std::uniform_real_distribution<float> sample(-1.f, 1.f);
    for (const int m : {1, 7, 8, 15, 16, 17, 23, 24, 31, 32, 33, 48, 64}) {
        for (const int n : {1, 3, 4, 5, 8, 9}) {
            for (const int k : {1, 7, 16, 33}) {
                const int lda = m + 3, ldb = k + 2, ldc = m + 5;
                std::vector<float> a(lda * k + 2, sentinel), b(ldb * n + 2, sentinel);
                std::vector<float> c(ldc * n + 2, sentinel);
                for (int t = 0; t < k; ++t)
                    for (int i = 0; i < m; ++i) a[1 + t * lda + i] = sample(rng);
                for (int j = 0; j < n; ++j)
                    for (int t = 0; t < k; ++t) b[1 + j * ldb + t] = sample(rng);
                const auto a_before = a, b_before = b;
                nss::gemm_nn_hwy(m, n, k, a.data() + 1, lda, b.data() + 1, ldb, c.data() + 1, ldc, true);
                if (a != a_before || b != b_before || c.front() != sentinel || c.back() != sentinel) return false;
                for (int j = 0; j < n; ++j) {
                    for (int i = 0; i < m; ++i) {
                        double expected = 0;
                        for (int t = 0; t < k; ++t)
                            expected += static_cast<double>(a[1 + t * lda + i]) * b[1 + j * ldb + t];
                        const float actual = c[1 + j * ldc + i];
                        if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-5) {
                            std::fprintf(stderr, "GemmNN m=%d n=%d k=%d i=%d j=%d mismatch %.9g / %.9g\n",
                                         m, n, k, i, j, actual, expected);
                            return false;
                        }
                    }
                    for (int i = m; i < ldc; ++i) if (c[1 + j * ldc + i] != sentinel) return false;
                }
            }
        }
    }
    return true;
}

bool test_dct16() {
    constexpr int side = 16, area = side * side;
    constexpr double pi = 3.1415926535897932384626433832795;
    double basis[side][side];
    for (int k = 0; k < side; ++k)
        for (int i = 0; i < side; ++i)
            basis[k][i] = std::sqrt((k == 0 ? 1.0 : 2.0) / side) * std::cos(pi * (i + 0.5) * k / side);
    std::mt19937 rng(2569);
    std::uniform_real_distribution<float> sample(-1.f, 1.f);
    for (int count : {1, 2, 3, 8, 16}) {
        std::vector<float> data(count * area + 2, sentinel);
        for (int p = 0; p < count; ++p)
            for (int i = 0; i < area; ++i)
                data[1 + p * area + i] = p % 3 == 0 ? sample(rng) : (p % 3 == 1 ? 1.f : (i == 31 ? 1.f : 0.f));
        const auto original = data;
        if (!nss::detail::dct16_2d_batch_fast(data.data() + 1, count, false)) {
            std::puts("Dct16 specialized target disabled; tested separately when bit 128 is enabled");
            return true;
        }
        for (int p = 0; p < count; ++p) {
            double tmp[area] = {};
            for (int y = 0; y < side; ++y)
                for (int u = 0; u < side; ++u)
                    for (int x = 0; x < side; ++x)
                        tmp[y * side + u] += original[1 + p * area + y * side + x] * basis[u][x];
            for (int v = 0; v < side; ++v)
                for (int u = 0; u < side; ++u) {
                    double expected = 0;
                    for (int y = 0; y < side; ++y) expected += tmp[y * side + u] * basis[v][y];
                    const float actual = data[1 + p * area + v * side + u];
                    if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-5) {
                        std::fprintf(stderr, "Dct16 count=%d patch=%d v=%d u=%d mismatch %.9g / %.9g\n",
                                     count, p, v, u, actual, expected);
                        return false;
                    }
                }
        }
        if (!nss::detail::dct16_2d_batch_fast(data.data() + 1, count, true)) return false;
        for (int i = 1; i <= count * area; ++i)
            if (!std::isfinite(data[i]) || std::abs(data[i] - original[i]) > 1e-5) return false;
        if (data.front() != sentinel || data.back() != sentinel) return false;
    }
    return true;
}
}  // namespace

int main() {
    if (!test_gemm() || !test_dct16()) return 1;
    std::puts("AVX2 DCT/Gemm numerical and guard checks passed");
    return 0;
}
