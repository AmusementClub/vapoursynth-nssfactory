// SPDX-License-Identifier: GPL-2.0-only
// BM3D collaborative filtering on the device. The math follows the CPU
// generic path (Bm3dFilterGroup): orthonormal 2D DCT per patch, orthonormal
// DCT along the zero-padded group axis of length `group`, then either the
// hard threshold (|c| < 2.7 sigma -> 0, DC kept, weight 1/kept) or the
// Wiener shrink against the reference cube (w = r^2/(r^2+sigma^2), DC 1,
// weight 1/sum w^2), inverse transforms, and per-group weighted aggregation.
// The CPU 8x8x8 FFTW-layout path is the same transform up to scaling.
#pragma once

#include "cuda/common/aggregate.hpp"
#include "cuda/common/match.hpp"

#include <cuda_runtime.h>

namespace nss_cuda {

// Loads the orthonormal DCT tables for the current device (idempotent, thread-safe).
void bm3d_init_tables(int device);

struct Bm3dGroupArgs {
    // Device arrays of per-frame plane pointers indexed by DeviceMatch::t
    // (one entry for spatial filtering). ref == nullptr: hard threshold.
    const float* const* src;    // noisy planes (filtered)
    const float* const* ref;    // reference planes for the Wiener stage
    int pitch;                  // floats, shared by every plane
    const DeviceMatch* matches; // batch * group (group-strided)
    const int* counts;          // batch
    int batch;
    int block;
    int group;
    float sigma;                // effective (profile-scaled) sigma
    float* values;              // batch * group * block^2, also the transform workspace
    float* ref_cube;            // batch * group * block^2 when ref != nullptr
    AggregatePatch* patches;    // batch * group; slice = match t, unused slots get -1
};
void bm3d_filter_groups(const Bm3dGroupArgs& args, cudaStream_t stream);

// out = den > 1e-12 ? num / den : src (nss::aggregate_finish).
void bm3d_finish(const float* num, const float* den, const float* src, int width, int height, int pitch, float* out,
                 cudaStream_t stream);

// acc_num += num, acc_den += den over `count` floats (rolling accumulation,
// one call per center in ascending center order as the CPU rolling path).
void accumulate_slice(float* acc_num, float* acc_den, const float* num, const float* den, std::size_t count,
                      cudaStream_t stream);

}  // namespace nss_cuda
