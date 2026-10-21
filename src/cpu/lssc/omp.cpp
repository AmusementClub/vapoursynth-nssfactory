// SPDX-License-Identifier: GPL-2.0-only
#include "cpu/lssc/mixed_omp.hpp"
// OMP screens correlations in FP32, then refines support selection in FP64.
// Its least-squares solve and residual retain double precision.
// Public dictionary/coefficients and the denoise/coding path remain FP32.
#include "nss/resources.hpp"
#include "nss/avx2_policy.hpp"
#include "nss/cpu_api.hpp"
#include "nss/cpu_common.hpp"
#include "nss/cpu_lssc.hpp"
#include "cpu/wnnm/jacobi8.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace nss {
namespace {
static_assert(sizeof(double) == 2 * sizeof(float));
// The API owns float scratch. memcpy accesses its object representation, so
// storing double residuals needs neither a stronger alignment nor aliasing a
// float object through a double lvalue. The advertised scratch bound already
// covers 2*m floats even for sparsity=1.
double residual_at(const float* scratch, int row) {
    double value;
    std::memcpy(&value, scratch + 2 * static_cast<std::size_t>(row), sizeof(value));
    return value;
}
void set_residual(float* scratch, int row, double value) {
    std::memcpy(scratch + 2 * static_cast<std::size_t>(row), &value, sizeof(value));
}
double dot_precise(const float* a, const float* b, int n) {
    double sum = 0;
    for (int i = 0; i < n; ++i) sum = std::fma(double(a[i]), double(b[i]), sum);
    return sum;
}
void solve_cholesky(int n, double* gram, double* rhs) {
    for (int i = 0; i < n; ++i) for (int j = 0; j <= i; ++j) {
        double value = gram[i * n + j];
        for (int p = 0; p < j; ++p) value = std::fma(-gram[i * n + p], gram[j * n + p], value);
        gram[i * n + j] = i == j ? std::sqrt(std::max(value, double(1e-12f))) : value / gram[j * n + j];
    }
    for (int i = 0; i < n; ++i) {
        double value = rhs[i];
        for (int p = 0; p < i; ++p) value = std::fma(-gram[i * n + p], rhs[p], value);
        rhs[i] = value / gram[i * n + i];
    }
    for (int i = n - 1; i >= 0; --i) {
        double value = rhs[i];
        for (int p = i + 1; p < n; ++p) value = std::fma(-gram[p * n + i], rhs[p], value);
        rhs[i] = value / gram[i * n + i];
    }
}
}

int lssc_omp_workspace(const float* y, int m, const float* D, int atoms, int ldd, int sparsity, float* a,
                       float* work, int work_floats) {
    if (!y || !D || !a || m < 1 || atoms < 1 || ldd < m) return 0;
    std::fill_n(a, atoms, 0.f);
    if (sparsity < 1 || !work || work_floats < lssc_omp_work_floats(m, atoms, sparsity)) return 0;
    const int kmax = std::min({8, atoms, sparsity});
    auto* used = reinterpret_cast<unsigned char*>(work + 2 * static_cast<std::size_t>(m));
    std::memset(used, 0, atoms);
    int support[8]{};
    double gram[64]{}, rhs[8]{};
    for (int row = 0; row < m; ++row) set_residual(work, row, y[row]);
    detail::MixedOmpSearch mixed_search(D, m, atoms, ldd);
    int count = 0;
    for (int iteration = 0; iteration < kmax; ++iteration) {
        int best = -1;
        double largest = 0;
        if (!mixed_search.select(work, used, best, largest)) {
            int atom = 0;
            // Independent atoms share a residual load. Each dot retains its row
            // order and explicit double FMA; support ties retain atom order.
            for (; atom + 8 <= atoms; atom += 8) {
                const float* c0 = D + static_cast<std::size_t>(atom) * ldd;
                const float* c1 = c0 + ldd;
                const float* c2 = c1 + ldd;
                const float* c3 = c2 + ldd;
                const float* c4 = c3 + ldd;
                const float* c5 = c4 + ldd;
                const float* c6 = c5 + ldd;
                const float* c7 = c6 + ldd;
                double sums[8]{};
                for (int row = 0; row < m; ++row) {
                    const double residual = residual_at(work, row);
                    sums[0] = std::fma(double(c0[row]), residual, sums[0]);
                    sums[1] = std::fma(double(c1[row]), residual, sums[1]);
                    sums[2] = std::fma(double(c2[row]), residual, sums[2]);
                    sums[3] = std::fma(double(c3[row]), residual, sums[3]);
                    sums[4] = std::fma(double(c4[row]), residual, sums[4]);
                    sums[5] = std::fma(double(c5[row]), residual, sums[5]);
                    sums[6] = std::fma(double(c6[row]), residual, sums[6]);
                    sums[7] = std::fma(double(c7[row]), residual, sums[7]);
                }
                for (int lane = 0; lane < 8; ++lane) {
                    if (used[atom + lane]) continue;
                    const double magnitude = std::abs(sums[lane]);
                    if (magnitude > largest) { largest = magnitude; best = atom + lane; }
                }
            }
            for (; atom < atoms; ++atom) {
                if (used[atom]) continue;
                const float* column = D + static_cast<std::size_t>(atom) * ldd;
                double correlation = 0;
                for (int row = 0; row < m; ++row)
                    correlation = std::fma(double(column[row]), residual_at(work, row), correlation);
                const double magnitude = std::abs(correlation);
                if (magnitude > largest) { largest = magnitude; best = atom; }
            }
        }
        if (best < 0 || largest < double(1e-12f)) break;
        used[best] = 1;
        support[count++] = best;
        for (int p = 0; p < count; ++p) {
            const float* dp = D + static_cast<std::size_t>(support[p]) * ldd;
            for (int q = p; q < count; ++q) {
                const double value = dot_precise(dp, D + static_cast<std::size_t>(support[q]) * ldd, m);
                gram[p * count + q] = gram[q * count + p] = value;
            }
            rhs[p] = dot_precise(dp, y, m);
        }
        solve_cholesky(count, gram, rhs);
        std::fill_n(a, atoms, 0.f);
        for (int p = 0; p < count; ++p) a[support[p]] = static_cast<float>(rhs[p]);
        double norm_squared = 0;
        for (int row = 0; row < m; ++row) {
            double residual = y[row];
            for (int p = 0; p < count; ++p)
                residual = std::fma(-rhs[p], double(D[row + static_cast<std::size_t>(support[p]) * ldd]), residual);
            set_residual(work, row, residual);
            norm_squared = std::fma(residual, residual, norm_squared);
        }
        if (norm_squared < double(1e-12f)) break;
    }
    return count;
}

int lssc_omp(const float* y, int m, const float* D, int atoms, int ldd, int sparsity, float* a) {
    if (!y || !D || !a || m < 1 || atoms < 1 || ldd < m) return 0;
    std::fill_n(a, atoms, 0.f);
    if (sparsity < 1) return 0;
    const int need = lssc_omp_work_floats(m, atoms, sparsity);
    float scratch[3008];
    if (need <= 3008) return lssc_omp_workspace(y, m, D, atoms, ldd, sparsity, a, scratch, 3008);
    ResourceVector<float> dynamic(need);
    return lssc_omp_workspace(y, m, D, atoms, ldd, sparsity, a, dynamic.data(), need);
}
}  // namespace nss
