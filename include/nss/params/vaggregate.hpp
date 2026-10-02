// SPDX-License-Identifier: GPL-2.0-only
#pragma once
// Resolved VAggregate arguments shared by every backend.

namespace nss {

struct VAggregateParams {
    int radius = 0;
    bool allow_legacy = false;
    int planes[3]{1, 1, 1};
};

}  // namespace nss
