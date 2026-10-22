// SPDX-License-Identifier: GPL-2.0-only
#pragma once
// BM4D CPU kernel (model in include/nss/params/bm4d.hpp).
#include <cstddef>

namespace nss {

// Work buffer size in floats for a volume of t frames of h x w.
std::size_t bm4d_work_floats(int t, int h, int w);

// Both stages on one volume: noisy and out are t x h x w, contiguous (frame,
// row, column), out may not alias noisy. sigma is in the volume's units
// (8-bit sigma / 255). basic, when not null, receives the hard stage's estimate.
// work needs no particular alignment (the kernel aligns it).
void bm4d_denoise_volume(const float* noisy, int t, int h, int w, float sigma, float* out, float* work,
                         std::size_t work_floats, float* basic = nullptr);

}  // namespace nss
