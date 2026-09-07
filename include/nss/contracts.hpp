#pragma once
#include <cstdint>

namespace nss {
inline constexpr int kSemanticVersion = 2;
inline constexpr int kContributionVersion = 2;
inline constexpr int kContributionNumDenSlices = 1;
enum class Model : int { BM3D = 1, WNNM, MCWNNM, TWSC, NCSR, NLH, LSSC, NLM };
enum class KernelStatus : int { Ok, Bypass, InvalidInput, NumericalFailure, Unsupported, ResourceFailure };
struct NoiseProfile {
    static constexpr int bm3d = 1;
    static constexpr float user_units = 255.f;
    static constexpr float bm3d_calibration = 0.75f;
    static constexpr float fused8_gain = 64.f;
    static float bm3d_effective(float user_sigma) noexcept {
        return (user_sigma * (1.f / user_units)) * bm3d_calibration;
    }
};
// Semantic identities, independent of backend storage handles or launch APIs.
struct MatchIdentity { int frame, x, y; std::uint64_t ordinal; };
struct GroupShape { int rows, actual_count, configured_count, channels; };
struct SearchQuery { int center_frame, plane, x, y, block; };
struct ContributionLayout { int version, radius, center, slice_layout, model, profile; };
struct FrameView {
    int frame, plane, width, height;
    std::int64_t stride_elements;
    float range_min, range_max;
};
} // namespace nss
