// SPDX-License-Identifier: GPL-2.0-only
#pragma once
// Temporal execution mode shared by every backend. radius == 0 is spatial;
// radius > 0 selects legacy (fat intermediate for VAggregate, the default for
// every backend) or rolling (chunked, returns normalized frames directly).

namespace nss {

enum class TemporalMode { Legacy, Rolling };

// How the two cache arguments are read.
// Fixed: rolling_cache_limit and rolling_cache_chunks name the same value (at
// most one may be given), the number of finished chunks kept.
// Adaptive: rolling_cache_chunks is the number kept at first (default 1) and
// rolling_cache_limit the number the cache may grow to (default 16) when
// chunks it dropped are asked for again. rolling_cache_chunks alone keeps the
// cache at that size.
enum class RollingCache { Fixed, Adaptive };

struct RollingParams {
    int rolling_chunk = 4;
    int cache_chunks = 1;  // finished chunks kept at first
    int cache_limit = 1;   // and at most; equal to cache_chunks unless the cache grows
};

}  // namespace nss
