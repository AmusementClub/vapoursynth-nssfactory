#pragma once

#include <algorithm>
#include "nss/cpu_common.hpp"
#include <bit>
#include <cmath>
#include <cstdint>

namespace nss::detail {

inline constexpr float kJacobiCorrelationTolerance = 1e-6f;
// Squared relative FP32 effective-rank cutoff; discard numerical null columns
// before normalizing them into spurious directions in downstream PCA.
inline constexpr float kSvdRankFloorSquared = 1e-14f;

// Integer classification remains valid in the fast-math translation units.
inline std::uint32_t svd_magnitude_bits(float value) noexcept {
    return float_classification_bits(value) & 0x7fffffffu;
}

inline float svd_input_scale(std::uint32_t maximum) noexcept {
    const float magnitude = std::bit_cast<float>(maximum);
    if (magnitude == 0.f || (magnitude >= 0x1p-8f && magnitude <= 0x1p8f)) return 1.f;
    int exponent = 0;
    std::frexp(magnitude, &exponent);
    return std::ldexp(1.f, std::clamp(-exponent, -120, 120));
}

// Scale the angle's two coordinates before squaring. The conventional zeta
// formula squares an unbounded ratio, producing Inf/NaN for near-null columns.
inline float jacobi_tangent(float app, float aqq, float apq) noexcept {
    const float delta = (aqq - app) * 0.5f;
    const float scale = std::max(std::fabs(delta), std::fabs(apq));
    const float x = delta / scale;
    const float y = apq / scale;
    return y / (x + std::copysign(std::sqrt(x * x + y * y), x));
}

}  // namespace nss::detail
