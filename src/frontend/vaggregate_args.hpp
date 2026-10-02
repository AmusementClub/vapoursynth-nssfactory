// SPDX-License-Identifier: GPL-2.0-only
#pragma once
#include <VapourSynth4.h>
#include "nss/params/vaggregate.hpp"

namespace nss::frontend {

inline constexpr const char* kVAggregateSignature =
    "clip:vnode;src:vnode;radius:int:opt;planes:int[]:opt;allow_legacy:int:opt;memory_limit_mb:int:opt;";

// radius (range checked later with the clips), planes and allow_legacy.
VAggregateParams parse_vaggregate(const VSAPI* vsapi, const VSMap* in, const VSVideoInfo& src, const char* ns);

// Clip format, radius range and fat-intermediate shape (height = src.height * (2r+1) * 2).
void validate_vaggregate_shape(const VSVideoInfo& src, const VSVideoInfo& fat, int radius, const char* ns);

}  // namespace nss::frontend
