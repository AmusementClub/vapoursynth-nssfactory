// SPDX-License-Identifier: GPL-2.0-only
#include "frontend/mcwnnm_args.hpp"
#include "frontend/args.hpp"
#include "frontend/validate.hpp"

namespace nss::frontend {

McwnnmParams parse_mcwnnm(const VSAPI* vsapi, const VSMap* in, const VSVideoInfo& vi, const char* ns) {
    McwnnmParams p;
    if (!is_const_32f(vi)) {
        fail(ns, "MCWNNM", "constant RGBS or YUV444PS required");
    }
    if (vi.format.numPlanes != 3 || vi.format.subSamplingW != 0 || vi.format.subSamplingH != 0 ||
        (vi.format.colorFamily != cfRGB && vi.format.colorFamily != cfYUV)) {
        fail(ns, "MCWNNM", "constant RGBS or YUV444PS required");
    }
    map_float_array(vsapi, in, "sigma", p.sigma, 3, kMcwnnmDefaultSigma);
    for (int c = 0; c < 3; ++c) {
        if (!finite_bits(p.sigma[c])) {
            fail(ns, "MCWNNM", "sigma must be finite");
        }
    }
    p.block_size = map_int(vsapi, in, "block_size", kMcwnnmDefaultBlock);
    p.block_step = map_int(vsapi, in, "block_step", kMcwnnmDefaultStep);
    p.group_size = map_int(vsapi, in, "group_size", kMcwnnmDefaultGroup);
    p.bm_range = map_int(vsapi, in, "bm_range", kMcwnnmDefaultRange);
    p.radius = map_int(vsapi, in, "radius", kWnnmDefaultRadius);
    p.ps_num = map_int(vsapi, in, "ps_num", kWnnmDefaultPsNum);
    p.ps_range = map_int(vsapi, in, "ps_range", kWnnmDefaultPsRange);
    p.residual = map_int(vsapi, in, "residual", kMcwnnmDefaultResidual);
    p.adaptive = map_int(vsapi, in, "adaptive_aggregation", kMcwnnmDefaultAdaptive);
    p.admm_iter = map_int(vsapi, in, "admm_iter", kMcwnnmDefaultAdmmIter);
    p.rho = map_float(vsapi, in, "rho", kMcwnnmDefaultRho);
    p.mu = map_float(vsapi, in, "mu", kMcwnnmDefaultMu);
    p.iters = map_int(vsapi, in, "iters", kMcwnnmDefaultIters);
    p.delta = map_float(vsapi, in, "delta", kMcwnnmDefaultDelta);
    if (p.block_size < 1 || p.block_size > kWnnmMaxBlock || p.group_size < 1 ||
        p.group_size > kWnnmMaxGroup || p.block_step < 1 || p.block_step > p.block_size || p.radius < 0 ||
        p.radius > kBmMaxRadius) {
        fail(ns, "MCWNNM", "invalid block_size/group_size/block_step/radius");
    }
    if (3 * p.block_size * p.block_size > kSvdMaxM) {
        fail(ns, "MCWNNM", "3*block_size*block_size exceeds SVD limit");
    }
    if (p.bm_range < 1 || p.bm_range > kBmMaxRange) {
        fail(ns, "MCWNNM", "bm_range must be in [1, 64]");
    }
    if (p.admm_iter < 1 || p.admm_iter > 1000 || !(p.rho > 0.f) || p.mu < 1.f ||
        !finite_bits(p.rho) || !finite_bits(p.mu)) {
        fail(ns, "MCWNNM", "invalid admm_iter/rho/mu (admm_iter in [1, 1000], mu >= 1)");
    }
    if (p.iters < 1 || p.iters > 64 || !finite_bits(p.delta) || p.delta < 0.f || p.delta > 1.f) {
        fail(ns, "MCWNNM", "invalid iters/delta (iters in [1, 64], delta in [0, 1])");
    }
    if (p.ps_num < 1 || p.ps_num > p.group_size || p.ps_range < 1 || p.ps_range > kBmMaxRange) {
        fail(ns, "MCWNNM", "invalid ps_num/ps_range");
    }
    return p;
}

}  // namespace nss::frontend
