#pragma once

// Min AVX2. Compile AVX2 + AVX3 + AVX3_ZEN4 (plan D2 / §6.2).
// SCALAR stays enabled: Highway requires a static fallback target; disabling it
// zeros HWY_ENABLED_BASELINE even with -mavx2. SSE and extra AVX-512 variants
// are dropped. Must be set before highway.h.
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS (HWY_SSE2 | HWY_SSSE3 | HWY_SSE4 | HWY_AVX3_DL | HWY_AVX3_SPR | HWY_AVX10_2)
#endif
