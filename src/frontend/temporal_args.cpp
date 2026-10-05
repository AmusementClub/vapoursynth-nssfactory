// SPDX-License-Identifier: GPL-2.0-only
#include "frontend/temporal_args.hpp"
#include "frontend/args.hpp"
#include "frontend/validate.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace nss::frontend {

TemporalMode parse_temporal_mode(const VSAPI* vsapi, const VSMap* in, int radius, const char* filter, const char* ns,
                                 TemporalMode unset) {
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
    if (radius <= 0) return TemporalMode::Legacy;
    if (mode_s.empty()) return unset;
    return mode_s == "rolling" ? TemporalMode::Rolling : TemporalMode::Legacy;
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

TemporalRequest parse_temporal(const VSAPI* vsapi, const VSMap* in, int default_radius, const char* filter,
                               const char* ns, TemporalMode unset) {
    const int radius = map_int(vsapi, in, "radius", default_radius);
    TemporalRequest request;
    request.rolling = parse_temporal_mode(vsapi, in, radius, filter, ns, unset) == TemporalMode::Rolling;
    if (request.rolling) request.params = parse_rolling(vsapi, in, radius, filter, ns);
    return request;
}

void aggregate_rolling(const VSAPI* vsapi, const VSMap* in, VSMap* out, VSCore* core, int default_radius,
                       const char* filter, const char* ns) {
    if (vsapi->mapGetError(out)) return;
    VSPlugin* plugin = vsapi->getPluginByNamespace(ns, core);
    if (!plugin) fail(ns, filter, "rolling mode could not find VAggregate");
    VSMap* args = vsapi->createMap();
    vsapi->mapConsumeNode(args, "clip", vsapi->mapGetNode(out, "clip", 0, nullptr), maReplace);
    vsapi->mapConsumeNode(args, "src", vsapi->mapGetNode(in, "clip", 0, nullptr), maReplace);
    vsapi->mapSetInt(args, "radius", map_int(vsapi, in, "radius", default_radius), maReplace);
    int device_err = 0;
    const int64_t device = vsapi->mapGetInt(in, "device_id", 0, &device_err);
    if (!device_err) vsapi->mapSetInt(args, "device_id", device, maReplace);
    VSMap* result = vsapi->invoke(plugin, "VAggregate", args);
    vsapi->freeMap(args);
    if (const char* error = vsapi->mapGetError(result)) {
        const std::string message = std::string(ns) + "." + filter + ": rolling aggregation failed: " + error;
        vsapi->freeMap(result);
        throw std::runtime_error(message);
    }
    vsapi->mapConsumeNode(out, "clip", vsapi->mapGetNode(result, "clip", 0, nullptr), maReplace);
    vsapi->freeMap(result);
}

void create_temporal(VSPublicFunction create, const VSMap* in, VSMap* out, VSCore* core, const VSAPI* vsapi,
                     int default_radius, const char* filter, const char* ns, TemporalMode unset) {
    const TemporalRequest request = parse_temporal(vsapi, in, default_radius, filter, ns, unset);
    create(in, out, nullptr, core, vsapi);
    if (request.rolling) aggregate_rolling(vsapi, in, out, core, default_radius, filter, ns);
}

}  // namespace nss::frontend
