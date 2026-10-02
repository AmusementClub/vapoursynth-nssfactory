// SPDX-License-Identifier: GPL-2.0-only
#pragma once
// Resolved NCSR arguments shared by every backend.
#include "nss/params.hpp"

namespace nss {

struct NcsrParams {
    float sigma[3]{kNcsrDefaultSigma, kNcsrDefaultSigma, kNcsrDefaultSigma};
    int block_size = kNcsrDefaultBlock;
    int block_step = kNcsrDefaultStep;
    int group_size = kNcsrDefaultGroup;
    int bm_range = kNcsrDefaultRange;
    int radius = kWnnmDefaultRadius;
    int ps_num = kWnnmDefaultPsNum;
    int ps_range = kWnnmDefaultPsRange;
    int iters = kNcsrDefaultIters;
    float delta = kNcsrDefaultDelta;
};

}  // namespace nss
