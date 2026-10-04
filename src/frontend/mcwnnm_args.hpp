// SPDX-License-Identifier: GPL-2.0-only
#pragma once
#include <VapourSynth4.h>
#include "nss/params/mcwnnm.hpp"
#include "frontend/temporal_args.hpp"

namespace nss::frontend {

inline constexpr const char* kMcwnnmSignature =
    "clip:vnode;sigma:float[]:opt;block_size:int:opt;block_step:int:opt;group_size:int:opt;"
    "bm_range:int:opt;radius:int:opt;ps_num:int:opt;ps_range:int:opt;residual:int:opt;"
    "adaptive_aggregation:int:opt;rclip:vnode:opt;admm_iter:int:opt;rho:float:opt;mu:float:opt;"
    "iters:int:opt;delta:float:opt;memory_limit_mb:int:opt;" NSS_TEMPORAL_SIGNATURE;

// Format and range checks; plane sizes are validated by the caller after rclip.
McwnnmParams parse_mcwnnm(const VSAPI* vsapi, const VSMap* in, const VSVideoInfo& vi, const char* ns);

}  // namespace nss::frontend
