// SPDX-License-Identifier: GPL-2.0-only
#pragma once
// NLH arguments, the frozen v6 presets and pure preset resolution, shared by
// every backend (no VapourSynth or kernel dependencies).
#include "nss/params.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace nss {

struct NlhImageOptions {
    // Zero integer fields and exactly -1 coefficient fields request a preset.
    // Public parsing permits these sentinels only through omitted arguments.
    std::array<int, 2> block{0, 0}, step{0, 0}, group{0, 0}, q{0, 0}, window{0, 0};
    int basic_iterations = 0, wiener_iterations = 0;
    int radius = 0, ps_num = 2, ps_range = 4;
    double basic_mix = -1, hard_strength = -1, wiener_sigma_scale = -1;
    bool real_noise = false;  // noise_model; selects nothing since model version 6
};

namespace nlh_detail {
// The one preset of model version 6; exact public fixtures are in
// tests/data/nlh_presets_v6.json. Version 5 chose among three by noise
// model and sigma (AWGN up to 50, AWGN above 50, real); version 6 keeps the
// first of them for every clip, with a search window of 24 / 16 instead of
// 40 / 40 and one Basic round fewer. Kernel math is unchanged.
inline constexpr NlhImageOptions kPreset{
    .block = {8, 16}, .step = {6, 15},
    .group = {16, 16}, .q = {4, 4}, .window = {24, 16},
    .basic_iterations = 3, .wiener_iterations = 2,
    .basic_mix = 0.6, .hard_strength = 1.0,
    .wiener_sigma_scale = 0.32,
};
}  // namespace nlh_detail

// Resolve omitted fields from the preset for the smallest selected plane
// dimension. The preset no longer depends on sigma (the largest selected one,
// in public 8-bit units) or on the noise model; both stay in the interface.
inline NlhImageOptions nlh_resolve_preset(double /*sigma*/, int available, const NlhImageOptions& requested) {
    if (available < 2) throw std::invalid_argument("nss: selected plane is smaller than block_size");
    const auto& preset = nlh_detail::kPreset;
    auto o = requested;
    for (int s = 0; s < 2; ++s) {
        o.block[s] = o.block[s] ? o.block[s] : std::min(preset.block[s], available);
        if (o.block[s] < 2 || o.block[s] > 16 || o.block[s] > available)
            throw std::invalid_argument("nss.NLH: invalid resolved block_size");
        o.step[s] = o.step[s] ? o.step[s] : std::min(preset.step[s], o.block[s]);
        if (!o.q[s]) {
            o.q[s] = preset.q[s];
            while (o.q[s] > o.block[s] * o.block[s]) o.q[s] /= 2;
        }
        o.group[s] = o.group[s] ? o.group[s] : preset.group[s];
        o.window[s] = o.window[s] ? o.window[s] : preset.window[s];
        if (o.q[s] < 2 || o.q[s] > 16 || (o.q[s] & (o.q[s] - 1)) ||
            o.q[s] > o.block[s] * o.block[s] || o.group[s] < 2 || o.group[s] > 64 ||
            (o.group[s] & (o.group[s] - 1)) || o.step[s] < 1 || o.step[s] > o.block[s] ||
            o.window[s] < 1 || o.window[s] > 129)
            throw std::invalid_argument("nss.NLH: invalid resolved stage shape");
    }
    o.basic_iterations = o.basic_iterations ? o.basic_iterations : preset.basic_iterations;
    o.wiener_iterations = o.wiener_iterations ? o.wiener_iterations : preset.wiener_iterations;
    if (o.basic_mix == -1) o.basic_mix = preset.basic_mix;
    if (o.hard_strength == -1) o.hard_strength = preset.hard_strength;
    if (o.wiener_sigma_scale == -1) o.wiener_sigma_scale = preset.wiener_sigma_scale;
    if (o.basic_iterations < 1 || o.basic_iterations > 64 || o.wiener_iterations < 1 || o.wiener_iterations > 64 ||
        !(o.basic_mix >= 0 && o.basic_mix <= 1) || !std::isfinite(o.basic_mix) ||
        !(o.hard_strength >= 0) || !std::isfinite(o.hard_strength) ||
        !(o.wiener_sigma_scale >= 0) || !std::isfinite(o.wiener_sigma_scale) ||
        o.radius < 0 || o.radius > 16 || o.ps_num < 1 || o.ps_num > std::min(o.group[0], o.group[1]) ||
        o.ps_range < 1 || o.ps_range > 64)
        throw std::invalid_argument("nss.NLH: invalid resolved iteration, coefficient or temporal parameters");
    return o;
}

// NLH processes RGB in its YUV working domain; propagate per-plane noise
// (normalized sigma and public units) through the same matrix.
inline void nlh_rgb_sigma_to_yuv(std::array<float, 3>& sigma, std::array<double, 3>& sigma_units) {
    constexpr double a[3][3]{{0.299, 0.587, 0.114}, {-0.168736607142857, -0.331263392857143, 0.5}, {0.5, -0.4186875, -0.0813125}};
    const auto in = sigma;
    const auto original_units = sigma_units;
    for (int c = 0; c < 3; ++c) {
        double variance = 0, units_variance = 0;
        for (int k = 0; k < 3; ++k) {
            variance += a[c][k] * a[c][k] * double(in[k]) * in[k];
            const double units = original_units[k] >= 0 ? original_units[k] : double(in[k]) * 255;
            units_variance += a[c][k] * a[c][k] * units * units;
        }
        sigma[c] = float(std::sqrt(variance));
        sigma_units[c] = std::sqrt(units_variance);
    }
}

}  // namespace nss
