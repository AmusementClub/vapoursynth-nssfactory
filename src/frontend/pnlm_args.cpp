// SPDX-License-Identifier: GPL-2.0-only
#include "frontend/pnlm_args.hpp"
#include "frontend/args.hpp"
#include "frontend/validate.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

namespace nss::frontend {

PnlmParams parse_pnlm(const VSAPI* vsapi, const VSMap* in, const VSVideoInfo& vi, const char* ns) {
    if (!is_const_32f(vi)) fail(ns, "PNLM", "constant Gray/YUV/RGB 32-bit float required");
    PnlmParams p;
    map_float_array(vsapi, in, "sigma", p.sigma, vi.format.numPlanes, kPnlmDefaultSigma);
    for (int plane = 0; plane < vi.format.numPlanes; ++plane) {
        const float s = p.sigma[plane] / 255.f;
        if (!(p.sigma[plane] >= 0.f) || !std::isfinite(p.sigma[plane]))
            fail(ns, "PNLM", "sigma must be finite and non-negative");
        if (p.sigma[plane] > 0.f && !(s * s >= std::numeric_limits<float>::min())) fail(ns, "PNLM", "sigma is too small");
    }
    p.d = map_int(vsapi, in, "d", kPnlmDefaultD);
    p.a = map_int(vsapi, in, "a", kPnlmDefaultA);
    p.block = map_int(vsapi, in, "block_size", kPnlmDefaultBlock);
    p.group = map_int(vsapi, in, "group_size", kPnlmDefaultGroup);
    if (p.d < 0 || p.d > kPnlmMaxD) fail(ns, "PNLM", "d must be in [0, 31]");
    if (p.a < 1 || p.a > kPnlmMaxA) fail(ns, "PNLM", "a must be in [1, 64]");
    if (p.block != 4 && p.block != 6 && p.block != 8) fail(ns, "PNLM", "block_size must be 4, 6 or 8");
    if (p.group < 2 || p.group > kPnlmMaxGroup) fail(ns, "PNLM", "group_size must be in [2, 32]");
    for (int plane = 0; plane < vi.format.numPlanes; ++plane) {
        if (p.sigma[plane] == 0.f) continue;
        const int w = plane_width(vi, plane), h = plane_height(vi, plane);
        if (w < p.block || h < p.block) fail(ns, "PNLM", "a filtered plane is smaller than block_size");
        if (w > kPnlmMaxSide || h > kPnlmMaxSide) fail(ns, "PNLM", "filtered planes are limited to 8192 x 8192");
    }
    // d pins up to 2d + 1 input frames outside the admission budget, as NLM's.
    const std::int64_t frame_bytes =
        static_cast<std::int64_t>(vi.format.bytesPerSample) * vi.width * vi.height * vi.format.numPlanes;
    if ((2LL * p.d + 1) * frame_bytes > kNlmMaxPinnedFrameBytes)
        fail(ns, "PNLM", "temporal radius d pins too much frame memory for this frame size");
    return p;
}

}  // namespace nss::frontend
