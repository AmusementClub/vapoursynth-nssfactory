// SPDX-License-Identifier: GPL-2.0-only
#include "frontend/bm4d_args.hpp"
#include "frontend/args.hpp"
#include "frontend/validate.hpp"

#include <cmath>
#include <limits>

namespace nss::frontend {

Bm4dParams parse_bm4d(const VSAPI* vsapi, const VSMap* in, const VSVideoInfo& vi, const char* ns) {
    if (!is_const_32f(vi)) fail(ns, "BM4D", "constant Gray/YUV/RGB 32-bit float required");
    if (vi.numFrames < 1) fail(ns, "BM4D", "a clip of known length required");
    Bm4dParams p;
    map_float_array(vsapi, in, "sigma", p.sigma, vi.format.numPlanes, kBm4dDefaultSigma);
    for (int plane = 0; plane < vi.format.numPlanes; ++plane) {
        const float s = p.sigma[plane] / 255.f;
        if (!(p.sigma[plane] >= 0.f) || !std::isfinite(p.sigma[plane])) fail(ns, "BM4D", "sigma must be finite and non-negative");
        if (p.sigma[plane] > 0.f && !(s * s >= std::numeric_limits<float>::min())) fail(ns, "BM4D", "sigma is too small");
    }
    return p;
}

}  // namespace nss::frontend
