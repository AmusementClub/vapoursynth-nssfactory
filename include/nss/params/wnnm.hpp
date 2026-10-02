// SPDX-License-Identifier: GPL-2.0-only
#pragma once
// Resolved WNNM arguments shared by every backend (no VapourSynth or kernel
// dependencies; includable from device code).
#include "nss/params.hpp"

namespace nss {

struct WnnmParams {
    float sigma[3]{kWnnmDefaultSigma, kWnnmDefaultSigma, kWnnmDefaultSigma};
    int block_size = kWnnmDefaultBlock;
    int block_step = kWnnmDefaultStep;
    int group_size = kWnnmDefaultGroup;
    int bm_range = kWnnmDefaultRange;
    int radius = kWnnmDefaultRadius;
    int ps_num = kWnnmDefaultPsNum;
    int ps_range = kWnnmDefaultPsRange;
    int residual = kWnnmDefaultResidual;
    int adaptive = kWnnmDefaultAdaptive;
};

}  // namespace nss
