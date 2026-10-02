// SPDX-License-Identifier: GPL-2.0-only
#pragma once
// Resolved TWSC/NLH (full-image) arguments shared by every backend.
#include "nss/params/nlh.hpp"
#include "nss/params/twsc.hpp"

#include <array>

namespace nss {

struct FullImageParams {
    std::array<float, 3> sigma{3, 3, 3};          // public 8-bit units, last value broadcast
    std::array<double, 3> sigma_units{3, 3, 3};   // exact public values when given
    bool estimate = false;                         // blind noise estimation per frame
    TwscImageOptions twsc;
    NlhImageOptions nlh;
};

}  // namespace nss
