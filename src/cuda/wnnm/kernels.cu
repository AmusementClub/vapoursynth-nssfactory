// SPDX-License-Identifier: GPL-2.0-only
#include "cuda/wnnm/kernels.hpp"
#include "cuda/common/gram_group.cuh"
#include "cuda/common/gram_shrink.cuh"

namespace nss_cuda {
namespace {

struct WnnmModel {
    float sigma;
    int residual;
    int adaptive;
    __device__ bool center() const { return residual != 0; }
    // The block kernels' finish (gram_shrink_spectrum with codes) on the
    // group's N threads: lane i forms row i of C, gain_i v[:, i]^T. The
    // spectrum is walked in descending order and stops at the first value
    // that falls to the threshold; s^2 - constant falls with s, so a value
    // is kept exactly when its place is below `start`, or below the rank
    // with s^2 - constant > 0, and `kept` is a count over the lanes.
    static constexpr int kLaneFinishMax = 16;
    template <int N>
    __device__ float finish_lane(float* g, const float* v, const DeviceMatch*, int n, int area, int lane) const {
        const float constant = 8.f * sqrtf(2.f * static_cast<float>(n)) * sigma * sigma;
        const int rank = min(area, n), start = min(residual ? 0 : 1, rank);
        float sv;
        const int place = singular_place<N>(g, lane, &sv);
        float gain = 0.f;
        bool kept = place < start;
        if (kept) {
            gain = 1.f;
        } else if (place < rank) {
            const float tmp = sv * sv - constant;
            if (tmp > 0.f) {
                gain = ((sv + sqrtf(tmp)) * 0.5f) / sv;
                kept = true;
            }
        }
        // The group's lanes: all 32, or one half of the warp at N = 16.
        const unsigned group_lanes = N == 32 ? 0xffffffffu : 0xffffu << (threadIdx.x & 16);
        const int count = __popc(__ballot_sync(0xffffffffu, kept) & group_lanes);
        __syncthreads();  // every eigenvalue is read before C overwrites g
#pragma unroll
        for (int j = 0; j < N; ++j) g[lane * N + j] = gain * v[j * N + lane];
        return adaptive && count > 0 ? 1.f / static_cast<float>(count) : 1.f;
    }

    template <int N>
    __device__ float finish(float* g, const float* v, const DeviceMatch*, int n, int area, bool codes) const {
        const float constant = 8.f * sqrtf(2.f * static_cast<float>(n)) * sigma * sigma;
        if (codes) {
            const int kept = gram_shrink_spectrum<N>(g, v, min(area, n), constant, residual ? 0 : 1, true);
            return adaptive && kept > 0 ? 1.f / static_cast<float>(kept) : 1.f;
        }
        // The solve kernel (g and v in registers): gram_shrink_spectrum's
        // gains with every index a constant (its order array would put them in
        // local memory). Value j is kept when its place in the descending
        // order is below `start`, or below the rank with s^2 - constant > 0,
        // as in finish_lane; M = V diag(gain) V^T in the same order.
        const int rank = min(area, n), start = min(residual ? 0 : 1, rank);
        float s[N], gain[N];
#pragma unroll
        for (int j = 0; j < N; ++j) {
            const float lambda = g[j * N + j];
            s[j] = lambda > 0.f ? sqrtf(lambda) : 0.f;
        }
        int kept = 0;
#pragma unroll
        for (int j = 0; j < N; ++j) {
            int place = 0;
#pragma unroll
            for (int i = 0; i < N; ++i) place += s[i] > s[j] || (s[i] == s[j] && i < j);
            bool keep = place < start;
            gain[j] = keep ? 1.f : 0.f;
            if (!keep && place < rank) {
                const float tmp = s[j] * s[j] - constant;
                if (tmp > 0.f) {
                    gain[j] = ((s[j] + sqrtf(tmp)) * 0.5f) / s[j];
                    keep = true;
                }
            }
            kept += keep;
        }
#pragma unroll
        for (int i = 0; i < N; ++i) {
#pragma unroll
            for (int j = 0; j < N; ++j) {
                float sum = 0.f;
#pragma unroll
                for (int k = 0; k < N; ++k) sum = fmaf(v[i * N + k] * gain[k], v[j * N + k], sum);
                g[i * N + j] = sum;
            }
        }
        return adaptive && kept > 0 ? 1.f / static_cast<float>(kept) : 1.f;
    }
};

}  // namespace

void wnnm_filter_groups(const WnnmGroupArgs& args, cudaStream_t stream) {
    const GramGroupArgs shell{args.src, args.pitch, args.matches, args.counts, args.batch,
                              args.block, args.group, args.values, args.patches, args.fused, args.scratch};
    launch_gram_groups(shell, WnnmModel{args.sigma, args.residual, args.adaptive}, stream);
}

}  // namespace nss_cuda
