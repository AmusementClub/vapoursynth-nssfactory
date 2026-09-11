#include "nss/backend.hpp"
#include "hwy/targets.h"
#include <cstdio>

int main() {
    const auto caps = nss::backend_caps();
#if HWY_ARCH_X86
    if (!(caps.runtime_targets & (HWY_AVX2 | HWY_AVX3 | HWY_AVX3_ZEN4))) {
        if (caps.executable || caps.selected_target || caps.float_lanes || nss::cpu_backend_available()) return 7;
        std::puts("No supported x86 ISA: host admission correctly rejected; native SIMD execution not tested.");
        return 0;
    }
#endif
    if (!caps.executable || caps.selected_target == 0 || caps.float_lanes < 1 ||
        !(caps.selected_target & caps.compiled_targets & caps.runtime_targets)) return 1;
#if HWY_ARCH_ARM_A64
    if (caps.compiled_targets & (HWY_ALL_SVE | HWY_NEON_BF16)) return 2;
    if (caps.portable_test) {
        if (caps.selected_target != HWY_EMU128) return 3;
    } else if (!(caps.selected_target & (HWY_NEON | HWY_NEON_WITHOUT_AES)) || caps.float_lanes != 4) {
        return 4;
    }
#endif
    // A supported static fallback must not accidentally admit pre-AVX2 x86,
    // nor an ARM production build with no native target. Never execute it.
    hwy::SetSupportedTargetsForTest(HWY_SCALAR);
    const auto unavailable = nss::backend_caps();
    hwy::SetSupportedTargetsForTest(0);
    if (unavailable.executable || unavailable.selected_target || unavailable.float_lanes) return 5;
    const auto restored = nss::backend_caps();
    if (!restored.executable || restored.selected_target != caps.selected_target) return 6;
    std::printf("backend=%s mode=%s lanes=%d compiled=%lld runtime=%lld\n",
                caps.target_name, caps.build_mode, caps.float_lanes,
                static_cast<long long>(caps.compiled_targets), static_cast<long long>(caps.runtime_targets));
    return 0;
}
