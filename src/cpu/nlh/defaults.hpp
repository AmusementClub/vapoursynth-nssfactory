#pragma once
#include "nss/cpu_image.hpp"

namespace nss::nlh_detail {
// Selected by the frozen development/selection procedure; exact public
// fixtures are in tests/data/nlh_presets_v5.json. Kernel math is unchanged.
inline constexpr NlhImageOptions kAwgnLow{
    .block = {8, 16}, .step = {6, 15},
    .group = {16, 16}, .q = {4, 4}, .window = {40, 40},
    .basic_iterations = 4, .wiener_iterations = 2,
    .basic_mix = 0.6, .hard_strength = 1.0,
    .wiener_sigma_scale = 0.32,
};
inline constexpr NlhImageOptions kAwgnHigh{
    .block = {8, 15}, .step = {6, 7},
    .group = {16, 16}, .q = {4, 4}, .window = {40, 40},
    .basic_iterations = 5, .wiener_iterations = 2,
    .basic_mix = 0.6, .hard_strength = 0.7071067811865476,
    .wiener_sigma_scale = 0.64,
};
inline constexpr NlhImageOptions kReal{
    .block = {7, 16}, .step = {4, 10},
    .group = {16, 16}, .q = {2, 4}, .window = {40, 40},
    .basic_iterations = 2, .wiener_iterations = 2,
    .basic_mix = 0.6, .hard_strength = 0.125,
    .wiener_sigma_scale = 0.64, .real_noise = true,
};
} // namespace nss::nlh_detail
