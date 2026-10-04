// SPDX-License-Identifier: GPL-2.0-only
// NLH on the device (the CPU nlh_filter_full / nlh_pass model). For a group of
// n matched patches (n a power of two) of m = block^2 pixels:
//   - pixel_match: every patch pixel row selects its q nearest rows of the
//     guide group (itself first, then by (distance, lowest index)),
//   - per channel, each row's q x n matrix of the selected pixels goes through
//     the 2D Haar transform, the Basic hard threshold or the Wiener gain, and
//     back,
//   - the results are summed per pixel into an m x n numerator with the
//     per-pixel selection counts as the denominator.
// The same matching/selection also drives the blind noise estimate.
#pragma once

#include "cuda/common/aggregate.hpp"
#include "cuda/common/match.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <limits>

namespace nss_cuda {

struct NlhGroupArgs {
    const float* const* guides;     // device array [t]: matching planes (pitch = width)
    const float* const* data;       // device array [t * nch + c]: filtered planes
    const float* const* reference;  // device array [t * nch + c]: Basic estimate (Wiener), else nullptr
    int width;                      // plane pitch in floats
    const DeviceMatch* matches;     // batch * group (group-strided)
    const int* counts;
    int batch;
    int block;
    int group;
    int q;
    int nch;
    bool pow2;                      // use the largest power of two <= count (filter passes)
    bool wiener;
    int wiener_iterations;
    float threshold[3];             // Basic: nlh_float_threshold(kNlhHardCoefficient * hard_strength * sigma_c)
    double noise[3];                // Wiener: (wiener_sigma_scale * sigma_c)^2
    bool identity[3];               // sigma_c == 0: the channel passes through
    // Workspace:
    float* guide_group;             // batch * m * group, [g][row][j]; unused with nlh_prepare_shared
    int* indices;                   // batch * m * q
    float* coef;                    // batch * nch * m * q * group; unused with nlh_filter_fused
    float* ref_coef;                // same; Wiener without nlh_local_matrix only
    float* values;                  // nch * batch * group * m, channel-major, patch-major inside
    float* den;                     // batch * m (shared by the group's patches and channels)
    AggregatePatch* patches;        // batch * group
};

// The float t with (|v| < t) == (double(|v|) < threshold) for every float v:
// the CPU compares in FP64, and the smallest float >= threshold decides the
// same way.
inline float nlh_float_threshold(double threshold) {
    const float t = static_cast<float>(threshold);
    return static_cast<double>(t) < threshold ? std::nextafter(t, std::numeric_limits<float>::infinity()) : t;
}

// Shared-memory budget of the per-group kernels.
inline constexpr std::size_t kNlhSharedBytes = 40 * 1024;
// The guide group (m x group) is staged in shared memory.
inline bool nlh_prepare_shared(int block, int group) {
    return static_cast<std::size_t>(block) * block * group * sizeof(float) <= kNlhSharedBytes;
}
// A row's q x n matrix fits a thread's local storage.
inline bool nlh_local_matrix(int group, int q) { return q * group <= 64 && group <= 32; }
// Bytes of shared memory of the fused filter: pixel indices and row matrices.
inline std::size_t nlh_fused_bytes(int block, int group, int q) {
    return static_cast<std::size_t>(block) * block * q * (group * sizeof(float) + sizeof(int));
}
inline bool nlh_filter_fused(int block, int group, int q) {
    return nlh_local_matrix(group, q) && nlh_fused_bytes(block, group, q) <= kNlhSharedBytes;
}

// Guide group, shared pixel indices and aggregation metadata.
void nlh_prepare_groups(const NlhGroupArgs& args, cudaStream_t stream);
// Haar shrinkage of every channel into values / den.
void nlh_filter_groups(const NlhGroupArgs& args, cudaStream_t stream);

// Blind noise estimate (nss::nlh_estimate_sigma): after nlh_prepare_groups
// with block 8, q 4 and pow2 = false, adds each group's statistic to
// totals[nch] (FP64). partial: batch * nch doubles; sums: (batch + 255) / 256
// * nch doubles.
void nlh_sigma_groups(const NlhGroupArgs& args, double* partial, double* sums, double* totals, cudaStream_t stream);

// Plane helpers (pitch = width).
void nlh_rgb_to_yuv(float* r, float* g, float* b, std::size_t count, bool luma_only, cudaStream_t stream);
void nlh_yuv_to_rgb(float* y, float* u, float* v, std::size_t count, cudaStream_t stream);
// basic = mix * basic + (1 - mix) * input.
void nlh_mix(float* basic, const float* input, std::size_t count, double mix, cudaStream_t stream);
// Area average of luma (width * sx by height * sy) into guide (width by height).
void nlh_area_guide(const float* luma, int luma_width, float* guide, int width, int height, int sx, int sy,
                    cudaStream_t stream);

}  // namespace nss_cuda
