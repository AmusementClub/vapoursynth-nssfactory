// SPDX-License-Identifier: GPL-2.0-only
#pragma once
#include <VapourSynth4.h>
#include "nss/params/lssc.hpp"

namespace nss::frontend {

inline constexpr const char* kLsscSignature =
    "clip:vnode;sigma:float[]:opt;block_size:int:opt;block_step:int:opt;radius:int:opt;memory_limit_mb:int:opt;";

// Format, shape and grid-size checks; plane sizes are validated by the caller.
LsscParams parse_lssc(const VSAPI* vsapi, const VSMap* in, const VSVideoInfo& vi, const char* ns);

}  // namespace nss::frontend
