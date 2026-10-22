// SPDX-License-Identifier: GPL-2.0-only
#pragma once
// DA3D CPU kernel (model in include/nss/params/da3d.hpp).
#include <cstddef>
#include <cstdint>

namespace nss {

struct Da3dImage {
    int channels = 1;  // 1, or 3 for RGB filtered jointly
    int width = 0;
    int height = 0;
    const float* noisy[3]{};
    const float* guide[3]{};
    float* out[3]{};
    std::ptrdiff_t noisy_stride[3]{};  // in floats
    std::ptrdiff_t guide_stride[3]{};
    std::ptrdiff_t out_stride[3]{};
    bool rgb = false;  // channels == 3: apply the orthonormal colour transform
    float sigma = 0.f; // noise standard deviation in the planes' units (8-bit sigma / 255)
};

struct Da3dStats {
    std::int64_t patches = 0;     // patches processed
    std::int64_t guide_only = 0;  // of which aggregated the guide (sum k below the mass)
};

// Work buffer size in floats for an image of this shape.
std::size_t da3d_work_floats(int width, int height, int channels);

// Filters img into img.out. work holds da3d_work_floats floats, 64-byte aligned.
void da3d_denoise(const Da3dImage& img, float* work, std::size_t work_floats, Da3dStats* stats = nullptr);

// The 64x64 complex DFT the kernel uses (exposed for tests): unnormalized,
// sign -1 when inverse is false and +1 when it is true; re and im are 4096
// floats each, row-major, transformed in place; scratch holds 8192 floats.
void da3d_dft64x64(float* re, float* im, bool inverse, float* scratch);

}  // namespace nss
