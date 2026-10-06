// SPDX-License-Identifier: GPL-2.0-only
#pragma once
#include <VapourSynth4.h>
#include "nss/params/temporal.hpp"

namespace nss::frontend {

// Case-insensitive temporal_mode; rolling only takes effect when radius > 0.
// `unset` is the backend's mode when the argument is absent or empty: legacy
// on the CPU, rolling on the device (where the fat intermediate costs a
// transfer several times the frame).
TemporalMode parse_temporal_mode(const VSAPI* vsapi, const VSMap* in, int radius, const char* filter, const char* ns,
                                 TemporalMode unset = TemporalMode::Legacy);

// Rolling chunk/cache arguments; only parsed once rolling mode is selected.
RollingParams parse_rolling(const VSAPI* vsapi, const VSMap* in, int radius, const char* filter, const char* ns,
                            RollingCache cache = RollingCache::Fixed);

// Arguments every temporal filter shares, appended to its own signature.
#define NSS_TEMPORAL_SIGNATURE \
    "temporal_mode:data:opt;rolling_chunk:int:opt;rolling_cache_chunks:int:opt;rolling_cache_limit:int:opt;"

// Both parsers above against the filter's own radius default. `rolling` is
// set only when the mode is rolling and radius > 0.
struct TemporalRequest {
    bool rolling = false;
    RollingParams params{};
};
TemporalRequest parse_temporal(const VSAPI* vsapi, const VSMap* in, int default_radius, const char* filter,
                               const char* ns, TemporalMode unset = TemporalMode::Legacy,
                               RollingCache cache = RollingCache::Fixed);

// Rolling as legacy + VAggregate: replaces the fat intermediate in out["clip"]
// with <ns>.VAggregate(fat, src=in["clip"], radius). Does nothing when `out`
// already carries an error.
void aggregate_rolling(const VSAPI* vsapi, const VSMap* in, VSMap* out, VSCore* core, int default_radius,
                       const char* filter, const char* ns);

// Runs `create`, then aggregates when rolling was requested. For filters whose
// rolling mode is legacy + VAggregate.
void create_temporal(VSPublicFunction create, const VSMap* in, VSMap* out, VSCore* core, const VSAPI* vsapi,
                     int default_radius, const char* filter, const char* ns,
                     TemporalMode unset = TemporalMode::Legacy, RollingCache cache = RollingCache::Fixed);

}  // namespace nss::frontend
