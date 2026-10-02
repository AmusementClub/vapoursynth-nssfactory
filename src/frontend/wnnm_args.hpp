// SPDX-License-Identifier: GPL-2.0-only
#pragma once
#include <VapourSynth4.h>
#include "nss/params/wnnm.hpp"

namespace nss::frontend {

// Shared argument list; a backend may append its own optional arguments.
inline constexpr const char* kWnnmSignature =
    "clip:vnode;sigma:float[]:opt;block_size:int:opt;block_step:int:opt;group_size:int:opt;"
    "bm_range:int:opt;radius:int:opt;ps_num:int:opt;ps_range:int:opt;residual:int:opt;"
    "adaptive_aggregation:int:opt;rclip:vnode:opt;memory_limit_mb:int:opt;";

// Format and range checks for the clip described by vi. Plane-size checks run
// after the caller has validated rclip (validate_group_planes).
WnnmParams parse_wnnm(const VSAPI* vsapi, const VSMap* in, const VSVideoInfo& vi, const char* ns);

}  // namespace nss::frontend
