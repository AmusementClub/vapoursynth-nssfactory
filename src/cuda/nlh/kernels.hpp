// SPDX-License-Identifier: GPL-2.0-only
// NLH on the device (the CPU nlh_filter_full / nlh_pass model). For a group of
// n matched patches (n a power of two) of m = block^2 pixels:
//   - pixel_match: every patch pixel row selects its q nearest rows of the
//     guide group (itself first, then by (distance, lowest index)),
//   - per channel, each row's q x n matrix of the selected pixels goes through
//     the 2D Haar transform, the Basic hard threshold or the Wiener gain, and
//     back,
//   - the results are summed per pixel into an m x n numerator with the
//     per-pixel selection counts as the denominator, and the kernel adds both
//     to the pass's accumulators (exact fixed-point sums by integer atomics).
// The blind noise estimate selects the same way in a kernel of its own.
#pragma once

#include "cuda/common/aggregate.hpp"
#include "cuda/common/match.hpp"

#include <cuda_runtime.h>

#include <algorithm>
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
    float noise_hi[3], noise_lo[3]; // noise as float(noise) and the remainder (nlh_set_noise)
    bool identity[3];               // sigma_c == 0: the channel passes through
    bool guided[3];                 // estimate: channel c's planes are the matching planes
    // Workspace:
    float* guide_group;             // batch * m * group, [g][row][j]; unused with nlh_prepare_shared
    int* indices;                   // batch * m * q
    float* coef;                    // batch * nch * m * q * group; unused with nlh_filter_fused
    float* ref_coef;                // same; Wiener without nlh_local_matrix only
    // Accumulators of the pass (the kernels aggregate their own output; see
    // FixedTarget for the units): the cell of a pixel of window slot t is
    // t * slice_step + y * width + x.
    unsigned long long* num[3];     // per channel: fixed-point sums of the pixel estimates
    unsigned* den;                  // selection counts (plain integers), shared by the channels
    std::size_t slice_step;
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
// Bytes of shared memory of a group's inverted selections: the (row, slot)
// list of every pixel and its cursor.
inline std::size_t nlh_lists_bytes(int block, int q) {
    const std::size_t m = static_cast<std::size_t>(block) * block;
    return m * q * sizeof(unsigned short) + m * sizeof(unsigned);
}
// Rows whose matrices the fused filter holds in shared memory at a time (0:
// the filter is not fused). A group of at most kNlhWholeBytes is held whole;
// a larger one goes through in equal chunks, at least two and none above
// kNlhChunkBytes: held whole it bounds the blocks an SM holds, while the
// rows outside a chunk wait for its shrink (measured at the presets: all 49
// rows for block 7, 32 of 64 for block 8, 128 of 256 for block 16).
inline constexpr std::size_t kNlhChunkBytes = 32 * 1024, kNlhWholeBytes = 8 * 1024;
inline int nlh_fused_rows(int block, int group, int q) {
    if (!nlh_local_matrix(group, q)) return 0;
    const std::size_t m = static_cast<std::size_t>(block) * block, row = q * group * sizeof(float);
    if (m * row <= kNlhWholeBytes) return static_cast<int>(m);
    const std::size_t chunks = std::max<std::size_t>(2, (m * row + kNlhChunkBytes - 1) / kNlhChunkBytes);
    return static_cast<int>((m + chunks - 1) / chunks);
}
// Bytes of shared memory of the fused filter: row matrices and the lists.
inline std::size_t nlh_fused_bytes(int block, int group, int q) {
    return static_cast<std::size_t>(nlh_fused_rows(block, group, q)) * q * group * sizeof(float) +
           nlh_lists_bytes(block, q);
}
inline bool nlh_filter_fused(int block, int group, int q) { return nlh_fused_rows(block, group, q) > 0; }

// The filter runs as the lane kernel (one thread per row slot, everything in
// registers, sums in fixed point) instead of the fused one: the shapes of
// the presets.
inline bool nlh_filter_lanes(int block, int group, int q) {
    return group == 16 && (q == 2 || q == 4) && q <= block * block && nlh_prepare_shared(block, group);
}
// Fixed point of the lane kernel's shared sums: a cell takes up to
// block * block values below the limit, and rows * limit * scale = 2^30.
inline float nlh_lane_limit(int block) { return block * block <= 64 ? 4.f : 2.f; }
inline float nlh_lane_scale(int block) { return block * block <= 64 ? 4194304.f : 2097152.f; }

// Sets the Wiener noise of channel c.
inline void nlh_set_noise(NlhGroupArgs& args, int c, double noise) {
    args.noise[c] = noise;
    args.noise_hi[c] = static_cast<float>(noise);
    args.noise_lo[c] = static_cast<float>(noise - static_cast<double>(args.noise_hi[c]));
}

// Guide group and shared pixel indices.
void nlh_prepare_groups(const NlhGroupArgs& args, cudaStream_t stream);
// Haar shrinkage of every channel, added to num / den.
void nlh_filter_groups(const NlhGroupArgs& args, cudaStream_t stream);
// The accumulators of `count` cells as float planes (num in sample units).
void nlh_finish_sums(const unsigned long long* num, const unsigned* den, float* out_num, float* out_den,
                     std::size_t count, cudaStream_t stream);

// Blind noise estimate (nss::nlh_estimate_sigma) of matched groups of block 8
// and group 16 (single frame): selects the pixels (q 4, every match used) and
// adds each group's statistic to totals[nch] (FP64). Reads guides, data,
// width, matches, counts, batch, nch and guided of args. partial: batch * nch
// doubles; sums: (batch + 255) / 256 * nch doubles.
void nlh_estimate_groups(const NlhGroupArgs& args, double* partial, double* sums, double* totals, cudaStream_t stream);

// Plane helpers (pitch = width).
void nlh_rgb_to_yuv(float* r, float* g, float* b, std::size_t count, bool luma_only, cudaStream_t stream);
void nlh_yuv_to_rgb(float* y, float* u, float* v, std::size_t count, cudaStream_t stream);
// basic = mix * basic + (1 - mix) * input.
void nlh_mix(float* basic, const float* input, std::size_t count, double mix, cudaStream_t stream);
// Area average of luma (width * sx by height * sy) into guide (width by height).
void nlh_area_guide(const float* luma, int luma_width, float* guide, int width, int height, int sx, int sy,
                    cudaStream_t stream);

}  // namespace nss_cuda
