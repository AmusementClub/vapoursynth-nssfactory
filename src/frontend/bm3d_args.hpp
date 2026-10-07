// SPDX-License-Identifier: GPL-2.0-only
#pragma once
#include <VapourSynth4.h>
#include "nss/params/bm3d.hpp"

namespace nss::frontend {

inline constexpr const char* kBm3dSignature =
    "clip:vnode;ref:vnode:opt;sigma:float[]:opt;block_size:int[]:opt;group_size:int[]:opt;"
    "block_step:int[]:opt;bm_range:int[]:opt;radius:int:opt;ps_num:int[]:opt;ps_range:int[]:opt;chroma:int:opt;"
    "final:int:opt;sigma_basic:float[]:opt;block_size_basic:int[]:opt;group_size_basic:int[]:opt;"
    "temporal_mode:data:opt;rolling_chunk:int:opt;rolling_cache_chunks:int:opt;rolling_cache_limit:int:opt;memory_limit_mb:int:opt;";

// Format, optional ref match (ref_vi may be null), per-plane array arguments
// with short-array inheritance, effective sigma, and shape checks.
Bm3dParams parse_bm3d(const VSAPI* vsapi, const VSMap* in, const VSVideoInfo& vi, const VSVideoInfo* ref_vi,
                      const char* ns);

// final=1 as two nodes of <ns>.BM3D: the basic estimate (the *_basic values
// in place of sigma, block_size and group_size; finished frames when
// radius > 0) and the call itself with that estimate as ref. `in` must have
// passed parse_bm3d. Puts the node in out["clip"].
void bm3d_two_nodes(const VSAPI* vsapi, const VSMap* in, VSMap* out, VSCore* core, const char* ns);

}  // namespace nss::frontend
