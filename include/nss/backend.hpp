#pragma once

#include <cstdint>

namespace nss {

// Target-level capability only. Kernel/shape coverage is tracked separately;
// a selected NEON target does not imply that every algorithm is accelerated.
struct BackendCaps {
    std::int64_t compiled_targets;
    std::int64_t runtime_targets;
    std::int64_t selected_target;
    const char* target_name;
    const char* build_mode;
    int float_lanes;
    bool executable;
    bool portable_test;
};

BackendCaps backend_caps() noexcept;
bool cpu_backend_available() noexcept;

}  // namespace nss
