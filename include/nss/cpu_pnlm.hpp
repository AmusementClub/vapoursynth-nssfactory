// SPDX-License-Identifier: GPL-2.0-only
#pragma once
// PNLM CPU kernel (model in include/nss/params/pnlm.hpp).
#include <cstddef>

namespace nss {

struct PnlmImage {
    int width = 0;
    int height = 0;
    int frames = 1;                          // the distinct frames searched, in clip order
    int center = 0;                          // the frame being filtered
    const float* const* src = nullptr;       // frames planes
    const std::ptrdiff_t* src_stride = nullptr;  // in floats
    float* out = nullptr;
    std::ptrdiff_t out_stride = 0;
    float sigma = 0.f;  // 8-bit units, > 0
    int a = 0;
    int block = 0;      // 4, 6 or 8
    int group = 0;      // 2 .. kPnlmMaxGroup
};

// Work buffer size in floats for a plane of this width.
std::size_t pnlm_work_floats(int width, int block);

// Filters img into img.out (width, height >= block). work holds
// pnlm_work_floats floats, 64-byte aligned.
void pnlm_denoise(const PnlmImage& img, float* work, std::size_t work_floats);

}  // namespace nss
