// SPDX-License-Identifier: GPL-2.0-only
#include "frontend/vaggregate_args.hpp"
#include "frontend/args.hpp"
#include "frontend/validate.hpp"
#include "nss/checked.hpp"
#include "nss/params.hpp"

#include <algorithm>

namespace nss::frontend {

VAggregateParams parse_vaggregate(const VSAPI* vsapi, const VSMap* in, const VSVideoInfo& src, const char* ns) {
    VAggregateParams p;
    p.radius = map_int(vsapi, in, "radius", 0);
    const int num_plane_args = vsapi->mapNumElements(in, "planes");
    if (num_plane_args > 0) {
        std::fill_n(p.planes, 3, 0);
        for (int i = 0; i < num_plane_args; ++i) {
            int err = 0;
            const int plane = static_cast<int>(vsapi->mapGetInt(in, "planes", i, &err));
            if (err || plane < 0 || plane >= src.format.numPlanes || p.planes[plane])
                fail(ns, "VAggregate", "planes must contain unique valid plane indices");
            p.planes[plane] = 1;
        }
    }
    const int allow_legacy = map_int(vsapi, in, "allow_legacy", 0);
    if (allow_legacy != 0 && allow_legacy != 1) fail(ns, "VAggregate", "allow_legacy must be 0 or 1");
    p.allow_legacy = allow_legacy != 0;
    return p;
}

void validate_vaggregate_shape(const VSVideoInfo& src, const VSVideoInfo& fat, int radius, const char* ns) {
    if (!is_const_32f(src) || !is_const_32f(fat)) fail(ns, "VAggregate", "constant 32f clips required");
    if (radius < 0 || radius > kBmMaxRadius) fail(ns, "VAggregate", "radius must be in [0, 16]");
    const int expect_h = checked_fat_height(src.height, radius);
    const VSVideoFormat& a = src.format;
    const VSVideoFormat& b = fat.format;
    const bool same_format = a.colorFamily == b.colorFamily && a.sampleType == b.sampleType &&
                             a.bitsPerSample == b.bitsPerSample && a.bytesPerSample == b.bytesPerSample &&
                             a.subSamplingW == b.subSamplingW && a.subSamplingH == b.subSamplingH &&
                             a.numPlanes == b.numPlanes;
    if (!same_format || fat.width != src.width || fat.height != expect_h || fat.numFrames != src.numFrames)
        fail(ns, "VAggregate",
             "clip must match src format, width, and frame count, with height=src.height*(2*radius+1)*2");
}

}  // namespace nss::frontend
