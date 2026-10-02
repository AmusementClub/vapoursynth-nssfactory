// SPDX-License-Identifier: GPL-2.0-only
#pragma once
// Resolved MCWNNM arguments shared by every backend.
#include "nss/params.hpp"

namespace nss {

struct McwnnmParams {
    float sigma[3]{kMcwnnmDefaultSigma, kMcwnnmDefaultSigma, kMcwnnmDefaultSigma};
    int block_size = kMcwnnmDefaultBlock;
    int block_step = kMcwnnmDefaultStep;
    int group_size = kMcwnnmDefaultGroup;
    int bm_range = kMcwnnmDefaultRange;
    int radius = kWnnmDefaultRadius;
    int ps_num = kWnnmDefaultPsNum;
    int ps_range = kWnnmDefaultPsRange;
    int residual = kMcwnnmDefaultResidual;
    int adaptive = kMcwnnmDefaultAdaptive;
    int admm_iter = kMcwnnmDefaultAdmmIter;
    float rho = kMcwnnmDefaultRho;
    float mu = kMcwnnmDefaultMu;
    int iters = kMcwnnmDefaultIters;
    float delta = kMcwnnmDefaultDelta;
};

}  // namespace nss
