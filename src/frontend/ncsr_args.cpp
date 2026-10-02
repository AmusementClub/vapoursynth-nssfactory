// SPDX-License-Identifier: GPL-2.0-only
#include "frontend/ncsr_args.hpp"
#include "frontend/args.hpp"
#include "frontend/validate.hpp"

#include <cmath>

namespace nss::frontend {

NcsrParams parse_ncsr(const VSAPI* vsapi, const VSMap* in, const VSVideoInfo& vi, const char* ns) {
    if (!is_const_32f(vi)) fail(ns, "NCSR", "constant Gray/YUV/RGB 32-bit float required");
    NcsrParams p;
    map_float_array(vsapi, in, "sigma", p.sigma, vi.format.numPlanes, kNcsrDefaultSigma);
    p.block_size = map_int(vsapi, in, "block_size", kNcsrDefaultBlock);
    p.block_step = map_int(vsapi, in, "block_step", kNcsrDefaultStep);
    p.group_size = map_int(vsapi, in, "group_size", kNcsrDefaultGroup);
    p.bm_range = map_int(vsapi, in, "bm_range", kNcsrDefaultRange);
    p.radius = map_int(vsapi, in, "radius", kWnnmDefaultRadius);
    p.ps_num = map_int(vsapi, in, "ps_num", kWnnmDefaultPsNum);
    p.ps_range = map_int(vsapi, in, "ps_range", kWnnmDefaultPsRange);
    p.iters = map_int(vsapi, in, "iters", kNcsrDefaultIters);
    p.delta = map_float(vsapi, in, "delta", kNcsrDefaultDelta);
    if (p.block_size < 1 || p.block_size > kWnnmMaxBlock || p.group_size < 1 || p.group_size > kWnnmMaxGroup ||
        p.block_step < 1 || p.block_step > p.block_size || p.radius < 0 || p.radius > kBmMaxRadius)
        fail(ns, "NCSR", "invalid block_size/group_size/block_step/radius");
    if (p.iters < 1 || p.iters > 64 || !std::isfinite(p.delta) || p.delta < 0.f || p.delta > 1.f)
        fail(ns, "NCSR", "invalid iters/delta (iters in [1, 64], delta in [0, 1])");
    if (p.ps_num < 1 || p.ps_num > p.group_size || p.ps_range < 1 || p.ps_range > kBmMaxRange)
        fail(ns, "NCSR", "invalid ps_num/ps_range");
    if (p.bm_range < 1 || p.bm_range > kBmMaxRange) fail(ns, "NCSR", "bm_range must be in [1, 64]");
    return p;
}

}  // namespace nss::frontend
