#include "nss/backend.hpp"
#include "cpu/hwy_config.hpp"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/backend.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
void ProbeBackend(std::int64_t* target, int* lanes) {
    *target = HWY_TARGET;
    *lanes = static_cast<int>(hwy::HWY_NAMESPACE::Lanes(hwy::HWY_NAMESPACE::ScalableTag<float>()));
}
}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(ProbeBackend);

BackendCaps backend_caps() noexcept {
    BackendCaps caps{HWY_TARGETS, hwy::SupportedTargets(), 0, "unavailable",
                     NSS_HWY_TARGET_MODE_STRING, 0, false, false};
    const auto available = caps.compiled_targets & caps.runtime_targets;
#if HWY_ARCH_ARM_A64
#if defined(HWY_COMPILE_ONLY_EMU128)
    caps.portable_test = true;
    // Reject compilers that silently replace EMU128 with scalar.
    caps.executable = (available & HWY_EMU128) != 0;
#else
    caps.executable = (available & (HWY_NEON | HWY_NEON_WITHOUT_AES)) != 0;
#endif
#elif HWY_ARCH_X86
    // Highway also checks OS vector-state support. Do not broaden the x86
    // support floor merely because a static fallback was compiled.
    caps.executable = (available & (HWY_AVX2 | HWY_AVX3 | HWY_AVX3_ZEN4)) != 0;
#endif
    if (caps.executable) {
        HWY_DYNAMIC_DISPATCH(ProbeBackend)(&caps.selected_target, &caps.float_lanes);
        caps.target_name = hwy::TargetName(caps.selected_target);
    }
    return caps;
}

bool cpu_backend_available() noexcept { return backend_caps().executable; }
}  // namespace nss
#endif
