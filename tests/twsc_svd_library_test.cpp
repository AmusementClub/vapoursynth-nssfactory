#include "cpu/twsc/svd_library.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

std::vector<float> fixture(int m, int n, int lda, double scale = 1) {
    std::vector<float> matrix(std::size_t(lda) * n, -17.0f);
    for (int j = 0; j < n; ++j) for (int i = 0; i < m; ++i)
        matrix[i + j * lda] = float(scale * (std::sin(.013 * (i + 1) * (j + 2)) +
                                            .07 * std::cos(.031 * (i + 3) * (j + 1))));
    return matrix;
}

std::vector<float> diagonal_fixture(int m, int n, int lda, bool repeated) {
    std::vector<float> matrix(std::size_t(lda) * n, 0.0f);
    for (int j = 0; j < n; ++j)
        for (int i = m; i < lda; ++i) matrix[i + j * lda] = -17.0f;
    const int rank = std::min(m, n);
    for (int k = 0; k < rank; ++k)
        matrix[k + k * lda] = repeated ? 1.0f : (k < 8 ? float(std::pow(.5, k)) : 0.0f);
    return matrix;
}
}

int main() {
    try {
        require(nss::twsc_svd_library_shape_eligible(192, 48, true, true), "admit default shape");
        require(!nss::twsc_svd_library_shape_eligible(191, 90, true, true), "reject short matrix");
        require(!nss::twsc_svd_library_shape_eligible(192, 47, true, true), "reject small group");
        require(nss::twsc_svd_library_shape_eligible(768, 256, true, true), "admit upper bounds");
        require(!nss::twsc_svd_library_shape_eligible(769, 256, true, true), "reject tall bound");
        require(!nss::twsc_svd_library_shape_eligible(768, 257, true, true), "reject wide bound");
        require(!nss::twsc_svd_library_shape_eligible(192, 48, false, true), "reject non-RGB layout");
        require(!nss::twsc_svd_library_shape_eligible(192, 48, true, false), "reject unequal noise");

        constexpr int m = 192, n = 48, lda = 197;
        auto matrix = fixture(m, n, lda);
        nss::TwscWorkspace work;
        auto status = nss::twsc_svd_library(matrix.data(), m, n, lda, work);
        require(status == nss::TwscSvdLibraryStatus::success, "DGESDD fixture");
        require(nss::twsc_valid_svd(matrix.data(), m, n, lda, work), "DGESDD validation");
        require(work.library_matrix.size() == std::size_t(m) * n, "matrix workspace");
        require(work.library_iwork.size() == std::size_t(8) * n, "integer workspace");
        require(matrix[m] == -17.0f && matrix[lda - 1] == -17.0f, "padding preserved");

        auto scaled = fixture(m, n, lda, 1e-20);
        nss::TwscWorkspace scaled_work;
        require(nss::twsc_svd_library(scaled.data(), m, n, lda, scaled_work) ==
                    nss::TwscSvdLibraryStatus::success,
                "scaled fixture");
        require(scaled_work.singular.front() > 0, "scaled spectrum");

        auto geometric = diagonal_fixture(m, n, lda, false);
        nss::TwscWorkspace geometric_work;
        require(nss::twsc_svd_library(geometric.data(), m, n, lda, geometric_work) ==
                    nss::TwscSvdLibraryStatus::success,
                "geometric spectrum");
        auto repeated = diagonal_fixture(m, n, lda, true);
        require(nss::twsc_svd_library(repeated.data(), m, n, lda, work) ==
                    nss::TwscSvdLibraryStatus::numerical_failure,
                "clustered spectrum fallback");
        auto rank_deficient = diagonal_fixture(m, n, lda, false);
        for (int k = 4; k < n; ++k) rank_deficient[k + k * lda] = 0;
        require(nss::twsc_svd_library(rank_deficient.data(), m, n, lda, work) ==
                    nss::TwscSvdLibraryStatus::success,
                "rank deficient spectrum");

        auto nonfinite = fixture(m, n, lda);
        nonfinite[3] = std::numeric_limits<float>::quiet_NaN();
        require(nss::twsc_svd_library(nonfinite.data(), m, n, lda, work) ==
                    nss::TwscSvdLibraryStatus::numerical_failure,
                "nonfinite input");

        std::vector<float> zero(std::size_t(lda) * n, 0.0f);
        require(nss::twsc_svd_library(zero.data(), m, n, lda, work) ==
                    nss::TwscSvdLibraryStatus::success,
                "zero matrix");
        for (double value : work.singular) require(value == 0, "zero spectrum");
        std::puts("TWSC DGESDD adapter checks passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
