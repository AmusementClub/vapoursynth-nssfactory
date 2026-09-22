#include "cpu/twsc/svd_library.hpp"

#include <lapacke.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace nss {
namespace {

static_assert(sizeof(lapack_int) == 4, "the SVD adapter requires LP64 lapack_int");
constexpr double kRankFloorSquared = 1e-14;

double matrix_scale(const float* a, int m, int n, int lda) {
    double maximum = 0;
    for (int j = 0; j < n; ++j) for (int i = 0; i < m; ++i) {
        const double value = a[i + j * lda];
        if (!std::isfinite(value)) return std::numeric_limits<double>::quiet_NaN();
        maximum = std::max(maximum, std::abs(value));
    }
    if (maximum == 0) return 1;
    int exponent = 0;
    std::frexp(maximum, &exponent);
    return std::ldexp(1.0, -exponent);
}

double scaled_rank_reference(const ResourceVector<double>& matrix, int m, int n) {
    double maximum = 0;
    if (m >= n) {
        for (int j = 0; j < n; ++j) {
            double norm = 0;
            for (int i = 0; i < m; ++i) norm += matrix[i + j * m] * matrix[i + j * m];
            maximum = std::max(maximum, norm);
        }
    } else {
        for (int i = 0; i < m; ++i) {
            double norm = 0;
            for (int j = 0; j < n; ++j) norm += matrix[i + j * m] * matrix[i + j * m];
            maximum = std::max(maximum, norm);
        }
    }
    return maximum;
}

bool finite_factors(const TwscWorkspace& work, int m, int n) {
    const int rank = std::min(m, n);
    for (double value : work.singular) if (!std::isfinite(value) || value < 0) return false;
    for (int k = 0; k < rank; ++k) {
        for (int i = 0; i < m; ++i) if (!std::isfinite(work.dictionary[i + k * m])) return false;
        for (int j = 0; j < n; ++j) if (!std::isfinite(work.vt[k + j * rank])) return false;
    }
    for (int k = 1; k < rank; ++k)
        if (work.singular[k] > work.singular[k - 1]) return false;
    return true;
}

} // namespace

const char* twsc_svd_library_status_name(TwscSvdLibraryStatus status) {
    switch (status) {
    case TwscSvdLibraryStatus::unavailable: return "unavailable";
    case TwscSvdLibraryStatus::success: return "success";
    case TwscSvdLibraryStatus::numerical_failure: return "numerical_failure";
    case TwscSvdLibraryStatus::resource_failure: return "resource_failure";
    case TwscSvdLibraryStatus::abi_failure: return "abi_failure";
    }
    return "unknown";
}

bool twsc_svd_library_shape_eligible(int m, int n, bool rgb_layout, bool uniform_row_sigma) {
    return rgb_layout && uniform_row_sigma && m >= 192 && n >= 48 && n <= kTwscMaxColumns &&
           m <= kTwscMaxRows;
}

TwscSvdLibraryStatus twsc_svd_library(const float* a, int m, int n, int lda,
                                      TwscWorkspace& work) {
    if (!a || m < 1 || m > kTwscMaxRows || n < 1 || n > kTwscMaxColumns || lda < m)
        return TwscSvdLibraryStatus::unavailable;
    const int rank = std::min(m, n);
    const double scale = matrix_scale(a, m, n, lda);
    if (!std::isfinite(scale) || scale <= 0) return TwscSvdLibraryStatus::numerical_failure;

    work.library_matrix.resize(std::size_t(m) * n);
    for (int j = 0; j < n; ++j) for (int i = 0; i < m; ++i)
        work.library_matrix[i + j * m] = double(a[i + j * lda]) * scale;
    const double rank_reference = scaled_rank_reference(work.library_matrix, m, n);
    const double rank_floor = rank_reference * kRankFloorSquared;
    work.dictionary.resize(std::size_t(m) * rank);
    work.singular.resize(rank);
    work.vt.resize(std::size_t(rank) * n);
    work.library_iwork.resize(std::size_t(8) * rank);
    work.library_work.resize(1);

    lapack_int lapack_m = static_cast<lapack_int>(m);
    lapack_int lapack_n = static_cast<lapack_int>(n);
    lapack_int lda_value = lapack_m;
    lapack_int ldu = lapack_m;
    lapack_int ldvt = static_cast<lapack_int>(rank);
    lapack_int query_lwork = -1;
    lapack_int info = LAPACKE_dgesdd_work(
        LAPACK_COL_MAJOR, 'S', lapack_m, lapack_n, work.library_matrix.data(), lda_value,
        work.singular.data(), work.dictionary.data(), ldu, work.vt.data(), ldvt,
        work.library_work.data(), query_lwork, work.library_iwork.data());
    if (info != 0 || !std::isfinite(work.library_work[0]) || work.library_work[0] < 1 ||
        work.library_work[0] > double(std::numeric_limits<lapack_int>::max()))
        return TwscSvdLibraryStatus::numerical_failure;

    const lapack_int lwork = static_cast<lapack_int>(work.library_work[0]);
    work.library_work.resize(static_cast<std::size_t>(lwork));
    // The query is allowed to overwrite A; copy the centered, scaled input
    // again before the actual factorization.
    for (int j = 0; j < n; ++j) for (int i = 0; i < m; ++i)
        work.library_matrix[i + j * m] = double(a[i + j * lda]) * scale;
    info = LAPACKE_dgesdd_work(
        LAPACK_COL_MAJOR, 'S', lapack_m, lapack_n, work.library_matrix.data(), lda_value,
        work.singular.data(), work.dictionary.data(), ldu, work.vt.data(), ldvt,
        work.library_work.data(), lwork, work.library_iwork.data());
    if (info != 0 || !finite_factors(work, m, n)) return TwscSvdLibraryStatus::numerical_failure;

    for (int k = 0; k < rank; ++k) {
        const double value = work.singular[k];
        work.singular[k] = value * value > rank_floor ? value / scale : 0;
        if (work.singular[k] == 0) {
            for (int i = 0; i < m; ++i) work.dictionary[i + k * m] = 0;
            for (int j = 0; j < n; ++j) work.vt[k + j * rank] = 0;
        }
    }
    if (!twsc_valid_svd(a, m, n, lda, work) || twsc_svd_clustered(work))
        return TwscSvdLibraryStatus::numerical_failure;
    return TwscSvdLibraryStatus::success;
}

} // namespace nss
