// SPDX-License-Identifier: GPL-2.0-only
#include "frontend/nlm_args.hpp"
#include "frontend/args.hpp"
#include "frontend/validate.hpp"

#include <cmath>
#include <cstdint>
#include <string>

namespace nss::frontend {

NlmParams parse_nlm(const VSAPI* vsapi, const VSMap* in, const VSVideoInfo& vi, const char* ns) {
    NlmParams p;
    if (!is_const_32f(vi)) {
        fail(ns, "NLM", "constant Gray/YUV/RGB 32-bit float required");
    }
    p.d = map_int(vsapi, in, "d", kNlmDefaultD);
    p.a = map_int(vsapi, in, "a", kNlmDefaultA);
    p.s = map_int(vsapi, in, "s", kNlmDefaultS);
    p.h = map_float(vsapi, in, "h", kNlmDefaultH);
    p.wref = map_float(vsapi, in, "wref", kNlmDefaultWref);
    if (p.d < 0 || p.d > kNlmMaxD || p.a <= 0 || p.s < 0 || p.a > kNlmMaxA ||
        p.s > kNlmMaxS ||
        !std::isfinite(p.h) || p.h <= 0.f || !std::isfinite(p.wref) || p.wref <= 0.f) {
        fail(ns, "NLM", "invalid d/a/s/h/wref");
    }
    // d pins 2d+1 input frames outside the admission budget; reject radii
    // whose pinned footprint is unreasonable for the actual frame size.
    const std::int64_t nlm_frame_bytes = static_cast<std::int64_t>(vi.format.bytesPerSample) * vi.width *
                                         vi.height * vi.format.numPlanes;
    if ((2LL * p.d + 1) * nlm_frame_bytes > kNlmMaxPinnedFrameBytes) {
        fail(ns, "NLM", "temporal radius d pins too much frame memory for this frame size");
    }
    const int wmode = map_int(vsapi, in, "wmode", 0);
    if (wmode != 0) {
        fail(ns, "NLM", "only wmode=0 (Welsch) is implemented");
    }
    int err = 0;
    const char* ch = vsapi->mapGetData(in, "channels", 0, &err);
    std::string cs = (!err && ch) ? ch : "AUTO";
    if (cs == "Y") {
        p.channels = ChannelMode::Y;
    } else if (cs == "UV") {
        p.channels = ChannelMode::UV;
    } else if (cs == "YUV") {
        p.channels = ChannelMode::YUV;
    } else if (cs == "RGB") {
        p.channels = ChannelMode::RGB;
    } else if (cs == "AUTO") {
        p.channels = (vi.format.colorFamily == cfRGB) ? ChannelMode::RGB : ChannelMode::Y;
    } else {
        fail(ns, "NLM", "channels must be Y, UV, YUV, RGB, or AUTO");
    }
    if (p.channels == ChannelMode::UV && vi.format.colorFamily != cfYUV) {
        fail(ns, "NLM", "UV requires YUV");
    }
    if (p.channels == ChannelMode::YUV &&
        (vi.format.colorFamily != cfYUV || vi.format.subSamplingW || vi.format.subSamplingH)) {
        fail(ns, "NLM", "YUV requires YUV444");
    }
    if (p.channels == ChannelMode::RGB && vi.format.colorFamily != cfRGB) {
        fail(ns, "NLM", "RGB requires RGB");
    }
    // Failure contract: the factory must validate the model's legal shape.
    // The distance/accumulation kernels clamp |ox| to w as defense-in-depth;
    // rejecting a >= width here keeps the documented band semantics exact
    // (|ox| >= plane width would otherwise write past the row into the next
    // row's accumulation state).
    int first_plane = 0;
    int last_plane = 0;
    if (p.channels == ChannelMode::UV) {
        first_plane = 1;
        last_plane = 2;
    } else if (p.channels != ChannelMode::Y) {
        last_plane = vi.format.numPlanes - 1;
    }
    for (int plane = first_plane; plane <= last_plane; ++plane) {
        if (p.a >= plane_width(vi, plane)) {
            fail(ns, "NLM", "search radius a must be smaller than the processed plane width");
        }
    }
    return p;
}

}  // namespace nss::frontend
