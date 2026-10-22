// SPDX-License-Identifier: GPL-2.0-only
#pragma once
#include <VapourSynth4.h>
#include "nss/params/bm4d.hpp"

namespace nss::frontend {

inline constexpr const char* kBm4dSignature = "clip:vnode;sigma:float[]:opt;memory_limit_mb:int:opt;";

// Format and sigma checks.
Bm4dParams parse_bm4d(const VSAPI* vsapi, const VSMap* in, const VSVideoInfo& vi, const char* ns);

}  // namespace nss::frontend
