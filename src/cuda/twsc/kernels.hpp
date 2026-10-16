// SPDX-License-Identifier: GPL-2.0-only
// TWSC group solve on the device (the CPU twsc_filter_full model). For a
// group of n matched patches with the nch channels stacked (m = nch * block^2
// rows):
//   - rows are centered; the SVD dictionary U and spectrum come from the Gram
//     matrix of the smaller side (tridiagonalization and implicit QL, one
//     thread block per group),
//   - the spectrum is reduced by the column noise, atoms = U diag(s') (not
//     formed with equal row weights),
//   - trilateral weighted sparse coding by ADMM: row weights 1 / sigma_row,
//     column weights from the per-patch residual noise; with equal row weights
//     the Sylvester system is diagonal, otherwise it is solved in the
//     eigenbasis of the weighted atom Gram matrix,
//   - reconstruction atoms C + mean, aggregated with weight 1 / column noise.
#pragma once

#include "cuda/common/aggregate.hpp"
#include "cuda/common/match.hpp"
#include "nss/params/twsc.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace nss_cuda {

inline constexpr int kTwscMaxFrames = 33;

struct TwscGroupArgs {
    const float* const* estimate;   // device array [t * nch + c]: current estimate planes
    const float* const* original;   // device array [t * nch + c]: input planes
    int width;                      // plane pitch in floats
    const DeviceMatch* matches;     // batch * group (group-strided)
    const int* counts;
    int batch;
    int block;
    int group;
    int nch;
    float sigma[kTwscMaxFrames * 3];  // [t * nch + c]: normalized noise of the matched frame's channel
    float row_sigma[3];             // per channel, of the reference frame
    bool residual;                  // rounds after the first: column noise from the estimate residual
    double lambda2;
    int iterations;
    double rho, mu, tolerance;
    float* centered;                // batch * m * group: centered group (column-major)
    void* work;                     // batch * twsc_work_reals(m, group) reals
    float* values;                  // nch * batch * group * block^2, channel-major
    AggregatePatch* patches;        // batch * group
    int* stalled;                   // device counter: groups that reached the iteration limit
    bool shared_matrices = false;   // set by twsc_filter_groups
    bool shared_values = false;     // set by twsc_filter_groups
    long long work_reals = 0;       // twsc_work_reals(m, group); set by twsc_filter_groups
};

// Reals of solver workspace per group.
__host__ __device__ inline std::size_t twsc_work_reals(int m, int group) {
    const std::size_t r = static_cast<std::size_t>(m < group ? m : group);
    return static_cast<std::size_t>(m) + 2 * group + 3 * r + 2 + 2 * r * r + m * r + 6 * r * group + 6 * r;
}
// Bytes of one real of the solver workspace (FP32: FP64 measured slower with no quality benefit).
std::size_t twsc_real_bytes();

// Groups whose blocks the device runs at a time. The blocks of a launch take
// about the same time, so they run in waves of this many and a batch that is
// not a multiple ends with a wave that leaves most of the device idle.
int twsc_resident_groups(int block, int group, int nch);

void twsc_filter_groups(const TwscGroupArgs& args, cudaStream_t stream);

// cfast (nss::twsc_color_split): three planes of `count` floats to the
// opponent planes in place, or back.
void twsc_opponent(float* a, float* b, float* c, std::size_t count, bool inverse, cudaStream_t stream);

}  // namespace nss_cuda
