// SPDX-License-Identifier: GPL-2.0-only
#pragma once
#include <VapourSynth4.h>
#include "nss/params/bm3d.hpp"

namespace nss::frontend {

inline constexpr const char* kBm3dSignature =
    "clip:vnode;ref:vnode:opt;sigma:float[]:opt;block_size:int[]:opt;group_size:int[]:opt;"
    "block_step:int[]:opt;bm_range:int[]:opt;radius:int:opt;ps_num:int[]:opt;ps_range:int[]:opt;"
    "temporal_mode:data:opt;rolling_chunk:int:opt;rolling_cache_chunks:int:opt;rolling_cache_limit:int:opt;memory_limit_mb:int:opt;";

// Format, optional ref match (ref_vi may be null), per-plane array arguments
// with short-array inheritance, effective sigma, and shape checks.
Bm3dParams parse_bm3d(const VSAPI* vsapi, const VSMap* in, const VSVideoInfo& vi, const VSVideoInfo* ref_vi,
                      const char* ns);

}  // namespace nss::frontend
