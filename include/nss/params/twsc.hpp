// SPDX-License-Identifier: GPL-2.0-only
#pragma once
// Resolved TWSC arguments shared by every backend.
#include "nss/params.hpp"

namespace nss {

struct TwscSolverOptions {
    int iterations = 10;
    double rho = 0.5;
    double mu = 1.1;
    double tolerance = 1e-6;
};

struct TwscImageOptions {
    int block = kTwscDefaultBlock, group = kTwscDefaultGroup, iterations = kTwscDefaultIters;
    int step = kTwscDefaultStep, window = 60, radius = 0, ps_num = 2, ps_range = 4;
    double lambda2 = 1, delta = 0;
    TwscSolverOptions solver;
    bool cfast = true;            // requested; see twsc_color_split
    bool planes_alone = false;    // every plane is a unit of its own (set by the node when cfast applies)
};

// cfast: an RGB frame whose three channels have one given sigma is filtered
// as three planes of an orthonormal opponent transform (white noise of equal
// sigma stays so), each plane alone, instead of as stacked three-channel
// patches. A third of the patch dimension, so a much smaller dictionary.
// Spatial frames only: the fat intermediate of radius > 0 has one
// denominator per plane and cannot be transformed back.
inline bool twsc_color_split(const TwscImageOptions& o, bool rgb, bool estimate, const double* sigma_units) {
    return o.cfast && rgb && !estimate && o.radius == 0 && sigma_units[0] > 0 && sigma_units[0] == sigma_units[1] &&
           sigma_units[0] == sigma_units[2];
}

// The transform and its inverse (the transpose), on one sample.
inline constexpr float kTwscOpponentA = 0.57735026918962576f;   // 1 / sqrt 3
inline constexpr float kTwscOpponentB = 0.70710678118654752f;   // 1 / sqrt 2
inline constexpr float kTwscOpponentC = 0.40824829046386302f;   // 1 / sqrt 6
template <class T>
constexpr void twsc_opponent_forward(T& r, T& g, T& b) {
    const T x = (r + g + b) * kTwscOpponentA, y = (r - b) * kTwscOpponentB, z = (r - 2 * g + b) * kTwscOpponentC;
    r = x, g = y, b = z;
}
template <class T>
constexpr void twsc_opponent_inverse(T& x, T& y, T& z) {
    const T base = x * kTwscOpponentA, r = base + y * kTwscOpponentB + z * kTwscOpponentC;
    const T g = base - 2 * (z * kTwscOpponentC), b = base - y * kTwscOpponentB + z * kTwscOpponentC;
    x = r, y = g, z = b;
}

}  // namespace nss
