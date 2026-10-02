// SPDX-License-Identifier: GPL-2.0-only
#include "frontend/lssc_args.hpp"
#include "frontend/args.hpp"
#include "frontend/validate.hpp"

namespace nss::frontend {

LsscParams parse_lssc(const VSAPI* vsapi, const VSMap* in, const VSVideoInfo& vi, const char* ns) {
    if (!is_const_32f(vi)) fail(ns, "LSSC", "constant Gray/YUV/RGB 32-bit float required");
    LsscParams p;
    map_float_array(vsapi, in, "sigma", p.sigma, vi.format.numPlanes, kLsscDefaultSigma);
    p.block_size = map_int(vsapi, in, "block_size", kLsscDefaultBlock);
    p.block_step = map_int(vsapi, in, "block_step", kLsscDefaultStep);
    p.radius = map_int(vsapi, in, "radius", 0);
    if (!lssc_allowed_block(p.block_size) || p.block_step < 1 || p.block_step > p.block_size)
        fail(ns, "LSSC", "block_size must be 1, 2, 4, 8, or 16");
    if (p.radius != 0) fail(ns, "LSSC", "radius>0 is not implemented (no temporal clustering)");
    const int patches = lssc_grid_count(vi.width, vi.height, p.block_size, p.block_step);
    const int area = p.block_size * p.block_size;
    if (area > 0 && patches > 16000000 / area) fail(ns, "LSSC", "grid too large; increase block_step");
    return p;
}

}  // namespace nss::frontend
