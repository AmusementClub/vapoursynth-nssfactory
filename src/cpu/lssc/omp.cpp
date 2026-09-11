#include "cpu/lssc/mixed_omp.hpp"
// OMP screens correlations in FP32, then refines support selection in FP64.
// Its least-squares solve and residual retain double precision.
// Public dictionary/coefficients and the denoise/ISTA path remain FP32.
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


void lssc_denoise_plane(const float* src, int width, int height, int sstride, float* num, float* den, int buf_stride,
                        int block, int step, float sigma, float* work, int work_floats) {
    if (!src || !num || !den || width < 1 || height < 1 || sstride < width || buf_stride < width) {
        return;
    }
    for (int y = 0; y < height; ++y) {
        std::memset(num + static_cast<std::size_t>(y) * static_cast<std::size_t>(buf_stride), 0,
                    static_cast<std::size_t>(width) * sizeof(float));
        std::memset(den + static_cast<std::size_t>(y) * static_cast<std::size_t>(buf_stride), 0,
                    static_cast<std::size_t>(width) * sizeof(float));
    }
    if (block < 1 || step < 1 || width < block || height < block) {
        return;
    }

    const int nx = lssc_axis_count(width, block, step);
    const int ny = lssc_axis_count(height, block, step);
    const int np = lssc_grid_count(width, height, block, step);
    if (np < 1) {
        return;
    }
    const int m = checked_int(static_cast<std::uint64_t>(block) * block);
    const int lda = m;
    const int atoms = std::min(kLsscDefaultAtoms, np);
    const int nclusters = std::min(kLsscDefaultClusters, np);
    const int assign_f =
        (static_cast<std::uint64_t>(np) * sizeof(int) + sizeof(float) - 1) / static_cast<int>(sizeof(float));
    const int counts_f = (nclusters * static_cast<int>(sizeof(int)) + static_cast<int>(sizeof(float)) - 1) /
                         static_cast<int>(sizeof(float));
    const int offsets_f = ((nclusters + 1) * static_cast<int>(sizeof(int)) + static_cast<int>(sizeof(float)) - 1) /
                          static_cast<int>(sizeof(float));
    const int ista_n = lssc_reconstruct_prepared_work_floats(m, np, atoms);
    const int cluster_n = lssc_cluster_work_floats(m, np, nclusters);
    const int dict_n = lssc_dict_work_floats(m, atoms, np, 1);
    const int work_need = lssc_denoise_work_floats(width, height, block, step);

    nss::ResourceVector<float> store;
    float* buf = work;
    if (!buf || work_floats < work_need) {
        store.assign(static_cast<std::size_t>(work_need), 0.f);
        buf = store.data();
    }
    float* patches = buf;
    float* D = patches + static_cast<std::size_t>(np) * static_cast<std::size_t>(lda);
    float* after_d = D + static_cast<std::size_t>(m) * static_cast<std::size_t>(atoms);
    int* assign = reinterpret_cast<int*>(after_d);
    float* after_assign = after_d + assign_f;
    int* counts = reinterpret_cast<int*>(after_assign);
    float* after_counts = after_assign + counts_f;
    int* offsets = reinterpret_cast<int*>(after_counts);
    float* after_offsets = after_counts + offsets_f;
    int* cursor = reinterpret_cast<int*>(after_offsets);
    float* after_cursor = after_offsets + counts_f;
    int* members = reinterpret_cast<int*>(after_cursor);
    float* cluster_work = after_cursor + assign_f;
    float* group = cluster_work + cluster_n;
    float* dict_work = group + static_cast<std::size_t>(np) * static_cast<std::size_t>(m);
    float* ista = dict_work + dict_n;
    float* prepare_work = ista + ista_n;
    std::memset(patches, 0, static_cast<std::size_t>(np) * static_cast<std::size_t>(lda) * sizeof(float));
    std::memset(D, 0, static_cast<std::size_t>(m) * static_cast<std::size_t>(atoms) * sizeof(float));

    int idx = 0;
    for (std::int64_t by0 = 0; by0 < static_cast<std::int64_t>(height) - block + step; by0 += step) {
        const int by = static_cast<int>(std::min<std::int64_t>(by0, std::max(0, height - block)));
        for (std::int64_t bx0 = 0; bx0 < static_cast<std::int64_t>(width) - block + step; bx0 += step) {
            const int bx = static_cast<int>(std::min<std::int64_t>(bx0, std::max(0, width - block)));
            pack_patch(patches + static_cast<std::size_t>(idx) * static_cast<std::size_t>(lda), lda, src, sstride, bx,
                       by, block, width, height);
            ++idx;
        }
    }
    const int n = idx;
    if (n < 1) {
        return;
    }

    lssc_cluster_workspace(patches, m, n, lda, nclusters, assign, counts, cluster_work, cluster_n);
    const bool avx2_gemm = NSS_AVX2_DEFAULTS && block == 8 && step == 8;
    lssc_dict_init_workspace(D, m, atoms, m, patches, n, lda, block, 1, 0x4C535343u, dict_work, dict_n, avx2_gemm);

    LsscPreparedContext prepared;
    if (lssc_prepare_context(D, m, atoms, m, prepare_work, lssc_prepare_work_floats(m, atoms), &prepared) != 0) {
        return;
    }
    prepared.avx2_gemm = avx2_gemm;
    offsets[0] = 0;
    for (int c = 0; c < nclusters; ++c) {
        offsets[c + 1] = offsets[c] + std::max(0, counts[c]);
        cursor[c] = offsets[c];
    }
    for (int j = 0; j < n; ++j) {
        const int c = (assign[j] >= 0 && assign[j] < nclusters) ? assign[j] : 0;
        members[cursor[c]++] = j;
    }

    for (int c = 0; c < nclusters; ++c) {
        const int cluster_begin = offsets[c];
        const int cluster_end = offsets[c + 1];
        const int nc = cluster_end - cluster_begin;
        if (nc < 1) {
            continue;
        }
        for (int g = 0; g < nc; ++g) {
            const int j = members[cluster_begin + g];
            std::memcpy(group + static_cast<std::size_t>(g) * static_cast<std::size_t>(m),
                        patches + static_cast<std::size_t>(j) * static_cast<std::size_t>(lda),
                        static_cast<std::size_t>(m) * sizeof(float));
        }
        lssc_reconstruct_prepared(group, m, nc, m, &prepared, sigma, ista, ista_n);
        for (int t = 0; t < nc; ++t) {
            const int j = members[cluster_begin + t];
            std::memcpy(patches + static_cast<std::size_t>(j) * static_cast<std::size_t>(lda),
                        group + static_cast<std::size_t>(t) * static_cast<std::size_t>(m),
                        static_cast<std::size_t>(m) * sizeof(float));
        }
    }

    for (int j = 0; j < n; ++j) {
        const int bx0 = (j % nx) * step;
        const int by0 = (j / nx) * step;
        const int bx = static_cast<int>(std::min<std::int64_t>(bx0, std::max(0, width - block)));
        const int by = static_cast<int>(std::min<std::int64_t>(by0, std::max(0, height - block)));
        const float* col = patches + static_cast<std::size_t>(j) * static_cast<std::size_t>(lda);
        for (int i = 0; i < m; ++i) {
            const float v = col[i];
            if (!is_finite_bits(v) || std::fabs(v) > 8.f) {
                pack_patch(patches + static_cast<std::size_t>(j) * static_cast<std::size_t>(lda), lda, src, sstride, bx,
                           by, block, width, height);
                break;
            }
        }
        aggregate_add(num, den, buf_stride, bx, by, col, block, width, height, 1.f);
    }
}

}  // namespace nss
