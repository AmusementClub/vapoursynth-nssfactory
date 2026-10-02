// SPDX-License-Identifier: GPL-2.0-only
#pragma once
#include <VapourSynth4.h>
#include "nss/params/nlm.hpp"

namespace nss::frontend {

inline constexpr const char* kNlmSignature =
    "clip:vnode;d:int:opt;a:int:opt;s:int:opt;h:float:opt;channels:data:opt;wmode:int:opt;"
    "wref:float:opt;rclip:vnode:opt;memory_limit_mb:int:opt;";

// Format, range, channel-mode and band checks (everything except rclip).
NlmParams parse_nlm(const VSAPI* vsapi, const VSMap* in, const VSVideoInfo& vi, const char* ns);

}  // namespace nss::frontend
