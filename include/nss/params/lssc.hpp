// SPDX-License-Identifier: GPL-2.0-only
#pragma once
// Resolved LSSC arguments and patch-grid geometry shared by every backend.
#include "nss/params.hpp"
#include "nss/checked.hpp"

#include <cstdint>

namespace nss {

struct LsscParams {
    float sigma[3]{kLsscDefaultSigma, kLsscDefaultSigma, kLsscDefaultSigma};
    int block_size = kLsscDefaultBlock;
    int block_step = kLsscDefaultStep;
    int radius = 0;
};

inline int lssc_axis_count(int len, int block, int step) {
    if (len < block || block < 1 || step < 1) {
        return 0;
    }
    return checked_int((static_cast<std::uint64_t>(len - block) + step - 1) / step + 1);
}

inline int lssc_grid_count(int width, int height, int block, int step) {
    return checked_int(static_cast<std::uint64_t>(lssc_axis_count(width, block, step)) * lssc_axis_count(height, block, step));
}

}  // namespace nss
