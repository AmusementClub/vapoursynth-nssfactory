// SPDX-License-Identifier: GPL-2.0-only
#include "frontend/temporal_args.hpp"
#include "frontend/args.hpp"
#include "frontend/validate.hpp"

#include <string>

namespace nss::frontend {

TemporalMode parse_temporal_mode(const VSAPI* vsapi, const VSMap* in, int radius, const char* filter, const char* ns) {
    int mode_err = 0;
    const char* mode = vsapi->mapGetData(in, "temporal_mode", 0, &mode_err);
    std::string mode_s;
    if (!mode_err && mode) {
        mode_s = mode;
        for (char& c : mode_s) {
            if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
        }
    }
    if (mode_s == "fused") fail(ns, filter, "temporal_mode=fused is not supported; use rolling or legacy");
    if (!mode_s.empty() && mode_s != "rolling" && mode_s != "legacy")
        fail(ns, filter, "temporal_mode must be rolling or legacy");
    return radius > 0 && mode_s == "rolling" ? TemporalMode::Rolling : TemporalMode::Legacy;
}

RollingParams parse_rolling(const VSAPI* vsapi, const VSMap* in, int radius, const char* filter, const char* ns) {
    RollingParams p;
    p.rolling_chunk = map_int(vsapi, in, "rolling_chunk", 4);
    int cache_limit_err = 0;
    p.cache_limit = map_int(vsapi, in, "rolling_cache_limit", 1, &cache_limit_err);
    int cache_chunks_err = 0;
    const int cache_chunks = map_int(vsapi, in, "rolling_cache_chunks", 1, &cache_chunks_err);
    if (!cache_chunks_err) p.cache_limit = cache_chunks;
    if (radius < 1) fail(ns, filter, "rolling mode requires radius > 0");
    if (p.rolling_chunk < 1 || p.rolling_chunk > 64) fail(ns, filter, "rolling_chunk must be in [1, 64]");
    if (!cache_limit_err && !cache_chunks_err)
        fail(ns, filter, "use only one of rolling_cache_limit and rolling_cache_chunks");
    if (p.cache_limit < 1 || p.cache_limit > 64) fail(ns, filter, "rolling_cache_limit must be in [1, 64]");
    return p;
}

}  // namespace nss::frontend
