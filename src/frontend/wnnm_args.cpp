// SPDX-License-Identifier: GPL-2.0-only
#include "frontend/wnnm_args.hpp"
#include "frontend/args.hpp"
#include "frontend/validate.hpp"

namespace nss::frontend {

WnnmParams parse_wnnm(const VSAPI* vsapi, const VSMap* in, const VSVideoInfo& vi, const char* ns) {
    if (!is_const_32f(vi)) fail(ns, "WNNM", "constant Gray/YUV/RGB 32-bit float required");
    WnnmParams p;
    map_float_array(vsapi, in, "sigma", p.sigma, vi.format.numPlanes, kWnnmDefaultSigma);
    p.block_size = map_int(vsapi, in, "block_size", kWnnmDefaultBlock);
    p.block_step = map_int(vsapi, in, "block_step", kWnnmDefaultStep);
    p.group_size = map_int(vsapi, in, "group_size", kWnnmDefaultGroup);
    p.bm_range = map_int(vsapi, in, "bm_range", kWnnmDefaultRange);
    p.radius = map_int(vsapi, in, "radius", kWnnmDefaultRadius);
    p.ps_num = map_int(vsapi, in, "ps_num", kWnnmDefaultPsNum);
    p.ps_range = map_int(vsapi, in, "ps_range", kWnnmDefaultPsRange);
    p.residual = map_int(vsapi, in, "residual", kWnnmDefaultResidual);
    p.adaptive = map_int(vsapi, in, "adaptive_aggregation", kWnnmDefaultAdaptive);
    if (p.block_size < 1 || p.block_size > kWnnmMaxBlock || p.group_size < 1 || p.group_size > kWnnmMaxGroup ||
        p.block_step < 1 || p.block_step > p.block_size || p.radius < 0 || p.radius > kBmMaxRadius)
        fail(ns, "WNNM", "invalid block_size/group_size/block_step/radius");
    if (p.bm_range < 1 || p.bm_range > kBmMaxRange) fail(ns, "WNNM", "bm_range must be in [1, 64]");
    if (p.ps_num < 1 || p.ps_num > p.group_size || p.ps_range < 1 || p.ps_range > kBmMaxRange)
        fail(ns, "WNNM", "invalid ps_num/ps_range");
    return p;
}

}  // namespace nss::frontend
