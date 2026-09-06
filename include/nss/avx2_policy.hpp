#pragma once

#ifndef NSS_AVX2_DEFAULTS
#define NSS_AVX2_DEFAULTS 0
#endif
#ifndef NSS_AVX2_REQUESTED
#define NSS_AVX2_REQUESTED 0
#endif

namespace nss::detail {
enum class Avx2Algorithm { BM3D, WNNM, TWSC, NCSR, NLH };

// Permissions are computed from the requested filter configuration before any
// group truncation. Unknown configurations use the original implementation.
inline unsigned avx2_policy(Avx2Algorithm algorithm, int block, int group,
                            int radius, bool wiener = false, int q = 0) noexcept {
#if NSS_AVX2_DEFAULTS
    unsigned result = 0;
    if (algorithm == Avx2Algorithm::BM3D) {
        if (radius == 0 && block == 16 && (group == 1 || group == 8 || group == 16)) result |= 128;
        if (radius == 0 && block == 12 && (group == 1 || group == 8 || group == 16 || group == 32)) result |= 512;
        if (radius == 0 && block == 8 && (group == 16 || group == 32 || group == 64)) result |= 1;
        if ((radius == 0 && (group == 8 || group == 16 || group == 32)) ||
            (radius == 1 && group == 16)) {
            if (block == 12) result |= 4;
            if (block == 16) result |= 8;
        }
    } else if (algorithm == Avx2Algorithm::WNNM && radius == 0) {
        if (block == 8 && group == 32) result |= 1;
        if (group == 16) {
            if (block == 12) result |= 4;
            if (block == 16) result |= 8;
        }
    } else if ((algorithm == Avx2Algorithm::TWSC || algorithm == Avx2Algorithm::NCSR) &&
               radius == 0 && block == 8 && group == 32) {
        result |= 1;
    } else if (algorithm == Avx2Algorithm::NLH) {
        if (radius == 1 && block == 8 && group == 16 && q == 4) result |= 1;
        if (q == 4 && group == 16 && ((block == 8 && (radius == 0 || radius == 1)) ||
                                    (block == 4 && radius == 0))) result |= 112;
    }
    return result;
#else
    return 0;
#endif
}
}  // namespace nss::detail
