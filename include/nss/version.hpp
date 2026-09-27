// SPDX-License-Identifier: GPL-2.0-only
#pragma once

#ifndef NSS_VERSION_STRING
#define NSS_VERSION_STRING "unknown"
#endif

namespace nss {
inline const char* version_string() {
    return NSS_VERSION_STRING;
}
}  // namespace nss
