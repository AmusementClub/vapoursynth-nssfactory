#pragma once

#include "nss/cpu_twsc_full.hpp"

namespace nss {

enum class TwscSvdLibraryStatus {
    unavailable,
    success,
    numerical_failure,
    resource_failure,
    abi_failure,
};

const char* twsc_svd_library_status_name(TwscSvdLibraryStatus status);

// The first admission predicate is intentionally independent of sigma and of
// the final image path. The caller supplies the color/layout and noise checks
// that are not represented by this low-level matrix adapter.
bool twsc_svd_library_shape_eligible(int m, int n, bool rgb_layout, bool uniform_row_sigma);

// Decompose the centered FP32 matrix into the existing thin TWSC workspace
// representation. The input is column-major with lda >= m and is unchanged.
// Clustered spectra are reported as numerical_failure so the caller can use
// the deterministic homemade FP64 route.
TwscSvdLibraryStatus twsc_svd_library(const float* a, int m, int n, int lda,
                                      TwscWorkspace& work);

} // namespace nss
