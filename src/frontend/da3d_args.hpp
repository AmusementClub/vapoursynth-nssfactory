// SPDX-License-Identifier: GPL-2.0-only
#pragma once
#include <VapourSynth4.h>
#include "nss/params/da3d.hpp"

namespace nss::frontend {

inline constexpr const char* kDa3dSignature = "clip:vnode;ref:vnode;sigma:float[]:opt;memory_limit_mb:int:opt;";

// Format and sigma checks; ref must have the clip's format, size and length
// (checked by the caller, which owns the nodes).
Da3dParams parse_da3d(const VSAPI* vsapi, const VSMap* in, const VSVideoInfo& vi, const char* ns);

// True when the three planes are filtered as one RGB image: RGB with one
// positive sigma for all three planes.
inline bool da3d_joint_rgb(const VSVideoInfo& vi, const Da3dParams& p) {
    return vi.format.colorFamily == cfRGB && p.sigma[0] > 0.f && p.sigma[0] == p.sigma[1] && p.sigma[1] == p.sigma[2];
}

}  // namespace nss::frontend
