// SPDX-License-Identifier: GPL-2.0-only
#include "frontend/da3d_args.hpp"
#include "frontend/args.hpp"
#include "frontend/validate.hpp"

#include <cmath>
#include <limits>

namespace nss::frontend {

Da3dParams parse_da3d(const VSAPI* vsapi, const VSMap* in, const VSVideoInfo& vi, const char* ns) {
    if (!is_const_32f(vi)) fail(ns, "DA3D", "constant Gray/YUV/RGB 32-bit float required");
    Da3dParams p;
    map_float_array(vsapi, in, "sigma", p.sigma, vi.format.numPlanes, kDa3dDefaultSigma);
    for (int plane = 0; plane < vi.format.numPlanes; ++plane)
    {
        const float s = p.sigma[plane] / 255.f;
        if (!(p.sigma[plane] >= 0.f) || !std::isfinite(p.sigma[plane])) fail(ns, "DA3D", "sigma must be finite and non-negative");
        if (p.sigma[plane] > 0.f && !(s * s >= std::numeric_limits<float>::min())) fail(ns, "DA3D", "sigma is too small");
    }
    return p;
}

}  // namespace nss::frontend
