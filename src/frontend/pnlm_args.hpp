// SPDX-License-Identifier: GPL-2.0-only
#pragma once
#include <VapourSynth4.h>
#include "nss/params/pnlm.hpp"

namespace nss::frontend {

inline constexpr const char* kPnlmSignature =
    "clip:vnode;sigma:float[]:opt;d:int:opt;a:int:opt;block_size:int:opt;group_size:int:opt;"
    "memory_limit_mb:int:opt;";

// Format, sigma, range and plane-size checks.
PnlmParams parse_pnlm(const VSAPI* vsapi, const VSMap* in, const VSVideoInfo& vi, const char* ns);

// The distinct frames n - d .. n + d of a clip of `frames` frames.
inline int pnlm_first_frame(int n, int d) { return n - d > 0 ? n - d : 0; }
inline int pnlm_last_frame(int n, int d, int frames) { return n + d < frames - 1 ? n + d : frames - 1; }

}  // namespace nss::frontend
