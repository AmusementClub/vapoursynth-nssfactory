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
    template <int N>
    __device__ float finish(float* g, const float* v, const DeviceMatch*, int n, int area, bool codes) const {
        const float constant = 8.f * sqrtf(2.f * static_cast<float>(n)) * sigma * sigma;
        const int kept = gram_shrink_spectrum<N>(g, v, min(area, n), constant, residual ? 0 : 1, codes);
        return adaptive && kept > 0 ? 1.f / static_cast<float>(kept) : 1.f;
    }
};

}  // namespace

void wnnm_filter_groups(const WnnmGroupArgs& args, cudaStream_t stream) {
    const GramGroupArgs shell{args.src, args.pitch, args.matches, args.counts, args.batch,
                              args.block, args.group, args.values, args.patches, args.fused};
    launch_gram_groups(shell, WnnmModel{args.sigma, args.residual, args.adaptive}, stream);
}

}  // namespace nss_cuda
