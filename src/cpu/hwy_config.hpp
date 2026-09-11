#pragma once

// CMake supplies the shared mask to CPU, qreplay, host and diagnostic targets.
// Standalone TUs follow the same minimum x86 / fixed-width ARM policy.
// SCALAR stays enabled: Highway requires a static fallback target; disabling it
// zeros HWY_ENABLED_BASELINE even with -mavx2. SSE and extra AVX-512 variants
// are dropped. Must be set before highway.h.
#ifndef HWY_DISABLED_TARGETS
#if defined(__aarch64__) || defined(_M_ARM64)
#define HWY_DISABLED_TARGETS (HWY_ALL_SVE | HWY_NEON_BF16)
#else
#define HWY_DISABLED_TARGETS (HWY_SSE2 | HWY_SSSE3 | HWY_SSE4 | HWY_AVX3_DL | HWY_AVX3_SPR | HWY_AVX10_2)
#endif
#endif
