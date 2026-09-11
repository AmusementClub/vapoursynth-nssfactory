#include "nss/cpu_lssc.hpp"
#include "cpu/lssc/gemm.hpp"
#include "cpu/wnnm/jacobi8.hpp"
#include "cpu/hwy_config.hpp"

#include <algorithm>
#include <cstdint>

#ifndef NSS_HAVE_LSSC_SME
#define NSS_HAVE_LSSC_SME 0
#endif
#if NSS_HAVE_LSSC_SME
#include <sys/sysctl.h>
extern "C" void nss_lssc_sme_product(int, int, int, const float*, int,
                                      const float*, float*, int);
#endif

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/lssc/gemm.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

void LsscPackColumns(const float* input, int rows, int columns, int stride, float* packed) {
    int column = 0;
#if HWY_MAX_BYTES >= 16
    const hn::FixedTag<float, 4> d;
    const hn::Repartition<std::uint64_t, decltype(d)> d64;
    for (; column + 4 <= columns; column += 4) {
        int row = 0;
        for (; row + 4 <= rows; row += 4) {
            const auto a = hn::LoadU(d, input + column * stride + row);
            const auto b = hn::LoadU(d, input + (column + 1) * stride + row);
            const auto c = hn::LoadU(d, input + (column + 2) * stride + row);
            const auto e = hn::LoadU(d, input + (column + 3) * stride + row);
            const auto ab0 = hn::BitCast(d64, hn::InterleaveLower(d, a, b));
            const auto ab1 = hn::BitCast(d64, hn::InterleaveUpper(d, a, b));
            const auto ce0 = hn::BitCast(d64, hn::InterleaveLower(d, c, e));
            const auto ce1 = hn::BitCast(d64, hn::InterleaveUpper(d, c, e));
            hn::StoreU(hn::BitCast(d, hn::InterleaveLower(d64, ab0, ce0)), d,
                       packed + row * columns + column);
            hn::StoreU(hn::BitCast(d, hn::InterleaveUpper(d64, ab0, ce0)), d,
                       packed + (row + 1) * columns + column);
            hn::StoreU(hn::BitCast(d, hn::InterleaveLower(d64, ab1, ce1)), d,
                       packed + (row + 2) * columns + column);
            hn::StoreU(hn::BitCast(d, hn::InterleaveUpper(d64, ab1, ce1)), d,
                       packed + (row + 3) * columns + column);
        }
        for (; row < rows; ++row)
            for (int lane = 0; lane < 4; ++lane)
                packed[row * columns + column + lane] = input[(column + lane) * stride + row];
    }
#endif
    for (; column < columns; ++column)
        for (int row = 0; row < rows; ++row)
            packed[row * columns + column] = input[column * stride + row];
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(LsscPackColumns);

bool lssc_sme_compiled() noexcept { return NSS_HAVE_LSSC_SME != 0; }

bool lssc_sme_available() noexcept {
#if NSS_HAVE_LSSC_SME
    static const bool available = [] {
        int value = 0;
        std::size_t size = sizeof(value);
        return sysctlbyname("hw.optional.arm.FEAT_SME", &value, &size, nullptr, 0) == 0 && value != 0;
    }();
    return available;
#else
    return false;
#endif
}

int lssc_gemm_pack_work_floats(int m, int n, int k) noexcept {
    // Initial measured domain: long products in b8/b16 LSSC reconstruction.
    // Keep short products and other shapes on the existing Highway path.
    if (m < 64 || m > 256 || k < 64 || k > 256 || m % 16 || k % 16 || n < 128 ||
        !lssc_sme_available()) return 0;
    return std::max(m, k) * std::min(n, 256);
}

void lssc_pack_columns_hwy(const float* input, int rows, int columns, int stride, float* packed) {
    HWY_DYNAMIC_DISPATCH(LsscPackColumns)(input, rows, columns, stride, packed);
}

bool lssc_gemm_nn(int m, int n, int k, const float* a, int lda,
                  const float* b, int ldb, float* c, int ldc,
                  float* scratch, int scratch_floats, bool avx2_gemm) {
#if NSS_HAVE_LSSC_SME
    const int need = lssc_gemm_pack_work_floats(m, n, k);
    if (need && scratch && scratch_floats >= need) {
        for (int first = 0; first < n;) {
            const int columns = std::min(n - first, 256);
            lssc_pack_columns_hwy(b + static_cast<std::size_t>(first) * ldb, k, columns, ldb, scratch);
            nss_lssc_sme_product(m, columns, k, a, lda, scratch,
                                 c + static_cast<std::size_t>(first) * ldc, ldc);
            first += columns;
        }
        return true;
    }
#else
    (void)scratch;
    (void)scratch_floats;
#endif
    gemm_nn_hwy(m, n, k, a, lda, b, ldb, c, ldc, avx2_gemm);
    return false;
}

}  // namespace nss
#endif
