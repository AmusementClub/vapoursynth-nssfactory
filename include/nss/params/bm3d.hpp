// SPDX-License-Identifier: GPL-2.0-only
#pragma once
// Resolved BM3D arguments shared by every backend. sigma holds the effective
// (profile-scaled) per-plane value; zero keeps a plane unprocessed. With
// chroma (CBM3D, YUV 4:4:4 only) the groups are matched on plane 0 and every
// plane is filtered with those groups: the search and shape arguments of
// plane 0 then hold for all three planes. With final the filter runs both
// stages: a basic estimate with the *_basic values (the rest as given), then
// the Wiener stage with that estimate as its reference, as
// BM3D(clip, ref=BM3D(clip, <basic values>), ...) does.
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
    bool chroma = false;
    bool final = false;
    float sigma_basic[3]{};  // effective, as sigma
    int block_size_basic[3]{};
    int group_size_basic[3]{};
    int block_step_basic[3]{};
    int ps_num_basic[3]{};
};

}  // namespace nss
