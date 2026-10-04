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
    // Set (num != nullptr) for shapes with bm3d_fuses(): the kernel adds the
    // weighted patches to these accumulators itself; values and patches are
    // then unused.
    FixedTarget fused{};
};

// Whether the kernel of this shape and stage can aggregate its own output
// (every shape whose cube is not transformed in `values`).
bool bm3d_fuses(int block, int group, bool wiener);
void bm3d_filter_groups(const Bm3dGroupArgs& args, cudaStream_t stream);

}  // namespace nss_cuda
