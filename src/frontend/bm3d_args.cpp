// SPDX-License-Identifier: GPL-2.0-only
#include "frontend/bm3d_args.hpp"
#include "frontend/args.hpp"
#include "frontend/validate.hpp"
#include "nss/contracts.hpp"

#include <algorithm>

namespace nss::frontend {

Bm3dParams parse_bm3d(const VSAPI* vsapi, const VSMap* in, const VSVideoInfo& vi, const VSVideoInfo* ref_vi,
                      const char* ns) {
    if (!is_const_32f(vi)) fail(ns, "BM3D", "constant Gray/YUV/RGB 32-bit float required");
    if (ref_vi && !same_video(vi, *ref_vi)) fail(ns, "BM3D", "ref must match clip");
    Bm3dParams p;
    const int np = vi.format.numPlanes;
    map_float_array(vsapi, in, "sigma", p.sigma, np, kBmDefaultSigma);
    for (int i = 0; i < np; ++i) {
        if (!finite_bits(p.sigma[i]) || p.sigma[i] < 0.f) fail(ns, "BM3D", "sigma must be finite and non-negative");
        if (p.sigma[i] != 0.f) p.sigma[i] = NoiseProfile::bm3d_effective(p.sigma[i]);
    }
    map_inherit_int(vsapi, in, "block_size", p.block_size, np, kBmBlock);
    map_inherit_int(vsapi, in, "group_size", p.group_size, np, kBmGroup);
    if (vsapi->mapNumElements(in, "block_step") <= 0) {
        for (int i = 0; i < np; ++i) p.block_step[i] = std::min(kBmDefaultStep, p.block_size[i]);
    } else {
        map_inherit_int(vsapi, in, "block_step", p.block_step, np, kBmDefaultStep);
    }
    map_int_array(vsapi, in, "bm_range", p.bm_range, np, kBmDefaultRange);
    if (vsapi->mapNumElements(in, "ps_num") <= 0) {
        for (int i = 0; i < np; ++i) p.ps_num[i] = std::min(kBmDefaultPsNum, p.group_size[i]);
    } else {
        map_inherit_int(vsapi, in, "ps_num", p.ps_num, np, kBmDefaultPsNum);
    }
    map_int_array(vsapi, in, "ps_range", p.ps_range, np, kBmDefaultPsRange);
    p.radius = map_int(vsapi, in, "radius", 0);
    if (p.radius < 0 || p.radius > kBmMaxRadius) fail(ns, "BM3D", "radius must be in [0, 16]");
    for (int i = 0; i < np; ++i) {
        if (!bm_allowed_block(p.block_size[i])) fail(ns, "BM3D", "block_size must be one of 1, 2, 4, 8, 12, 16, 32");
        if (!bm_allowed_group(p.group_size[i])) fail(ns, "BM3D", "group_size must be one of 1, 2, 4, 8, 16, 32, 64");
        if (p.block_size[i] > plane_width(vi, i) || p.block_size[i] > plane_height(vi, i))
            fail(ns, "BM3D", "block_size must not exceed plane dimensions");
        if (p.block_step[i] < 1 || p.block_step[i] > p.block_size[i])
            fail(ns, "BM3D", "block_step must be in [1, block_size]");
        if (p.ps_num[i] < 1 || p.ps_num[i] > p.group_size[i]) fail(ns, "BM3D", "ps_num must be in [1, group_size]");
        if (p.bm_range[i] < 1 || p.bm_range[i] > kBmMaxRange) fail(ns, "BM3D", "bm_range must be in [1, 64]");
        if (p.ps_range[i] < 0 || p.ps_range[i] > kBmMaxRange) fail(ns, "BM3D", "ps_range must be in [0, 64]");
    }
    return p;
}

}  // namespace nss::frontend
