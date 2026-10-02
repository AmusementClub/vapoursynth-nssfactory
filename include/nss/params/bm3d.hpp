// SPDX-License-Identifier: GPL-2.0-only
#pragma once
// Resolved BM3D arguments shared by every backend. sigma holds the effective
// (profile-scaled) per-plane value; zero keeps a plane unprocessed.
#include "nss/params.hpp"

namespace nss {

struct Bm3dParams {
    float sigma[3]{kBmDefaultSigma, kBmDefaultSigma, kBmDefaultSigma};
    int block_size[3]{kBmBlock, kBmBlock, kBmBlock};
    int group_size[3]{kBmGroup, kBmGroup, kBmGroup};
    int block_step[3]{kBmDefaultStep, kBmDefaultStep, kBmDefaultStep};
    int bm_range[3]{kBmDefaultRange, kBmDefaultRange, kBmDefaultRange};
    int ps_num[3]{kBmDefaultPsNum, kBmDefaultPsNum, kBmDefaultPsNum};
    int ps_range[3]{kBmDefaultPsRange, kBmDefaultPsRange, kBmDefaultPsRange};
    int radius = 0;
};

}  // namespace nss
