// SPDX-License-Identifier: GPL-2.0-only
#pragma once
// Temporal execution mode shared by every backend. radius == 0 is spatial;
// radius > 0 selects legacy (fat intermediate for VAggregate, the default for
// every backend) or rolling (chunked, returns normalized frames directly).

namespace nss {

enum class TemporalMode { Legacy, Rolling };

struct RollingParams {
    int rolling_chunk = 4;
    int cache_limit = 1;
};

}  // namespace nss
