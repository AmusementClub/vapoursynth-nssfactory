// SPDX-License-Identifier: GPL-2.0-only
// PNLM on the device (model in include/nss/params/pnlm.hpp; the same bits as
// the CPU kernel). Host-callable launchers (plain C++ header).
#pragma once
#include "nss/params/pnlm.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace nss_cuda {

inline constexpr int kPnlmMaxFrames = 2 * nss::kPnlmMaxD + 1;

// One plane: the frames searched (device planes, pitch in floats), the
// reference's at center.
struct PnlmPlane {
    const float* frames[kPnlmMaxFrames];
    int count;
    int center;
    int width, height, pitch;
    nss::PnlmScale scale;
    int a, block, group;
};

// Scratch of a plane of this size: the estimates of every reference patch,
// and the match keys and counts (when pnlm_fused does not take it).
std::size_t pnlm_est_bytes(int width, int height, int block);
std::size_t pnlm_keys_bytes(int width, int height, int block, int group);
// True when the matcher weights its references itself (no key list): small
// windows (a <= 4) and groups of up to 16.
bool pnlm_fused(int a, int group);

// Filters the plane into out (pitch floats a row).
void pnlm_run(const PnlmPlane& plane, float* est, void* keys, float* out, cudaStream_t stream);

}  // namespace nss_cuda
