#pragma once

#include "nss/cpu_common.hpp"

namespace nss::detail {

// SVD returns singular magnitudes even for an indefinite symmetric matrix.
// Above the existing roundoff band, a PSD mode must have agreeing left/right
// singular vectors. Tiny modes remain covered by the uncertainty/QR policy.
inline bool gram_spectrum_is_psd8(const float* u, const float* singular,
                                  const float* v, float uncertainty) {
    for (int k = 0; k < 8; ++k) {
        if (!(singular[k] > uncertainty)) continue;
        const float* a = u + k * 8;
        const float* b = v + k * 8;
        const float s0 = a[0] * b[0] + a[1] * b[1];
        const float s1 = a[2] * b[2] + a[3] * b[3];
        const float s2 = a[4] * b[4] + a[5] * b[5];
        const float s3 = a[6] * b[6] + a[7] * b[7];
        const float agreement = (s0 + s1) + (s2 + s3);
        if (!is_finite_bits(agreement) || agreement < 0.f) return false;
    }
    return true;
}

}  // namespace nss::detail
