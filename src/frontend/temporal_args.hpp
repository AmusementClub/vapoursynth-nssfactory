// SPDX-License-Identifier: GPL-2.0-only
#pragma once
#include <VapourSynth4.h>
#include "nss/params/temporal.hpp"

namespace nss::frontend {

// Case-insensitive temporal_mode; rolling only takes effect when radius > 0.
TemporalMode parse_temporal_mode(const VSAPI* vsapi, const VSMap* in, int radius, const char* filter, const char* ns);

// Rolling chunk/cache arguments; only parsed once rolling mode is selected.
RollingParams parse_rolling(const VSAPI* vsapi, const VSMap* in, int radius, const char* filter, const char* ns);

}  // namespace nss::frontend
