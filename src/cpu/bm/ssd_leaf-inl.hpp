// SPDX-License-Identifier: GPL-2.0-only
// Intentionally re-included inside each Highway target namespace (no include
// guard). ssd.cpp (ssd_block) and common/image_ssd_row.cpp (row kernel) must
// produce bit-identical block distances, so both use these exact leaf
// expression trees instead of keeping hand-mirrored copies.

#if HWY_MAX_BYTES >= 32
static float HSum8(hn::Vec<hn::FixedTag<float, 8>> v) {
    const hn::FixedTag<float, 4> d4;
    return hn::ReduceSum(d4, hn::Add(hn::LowerHalf(d4, v), hn::UpperHalf(d4, v)));
}
#endif

#if HWY_MAX_BYTES >= 16
static float Ssd4(const float* a, int sa, const float* b, int sb) {
    const hn::FixedTag<float, 4> d;
    auto acc = hn::Zero(d);
    for (int y = 0; y < 4; ++y) {
        const auto diff = hn::Sub(hn::LoadU(d, a + y * sa), hn::LoadU(d, b + y * sb));
        acc = hn::MulAdd(diff, diff, acc);
    }
    return hn::ReduceSum(d, acc);
}
#endif

static float Ssd8(const float* a, int sa, const float* b, int sb) {
#if HWY_MAX_BYTES >= 32
    const hn::FixedTag<float, 8> d;
    auto acc = hn::Zero(d);
    for (int y = 0; y < 8; ++y) {
        const auto diff = hn::Sub(hn::LoadU(d, a + y * sa), hn::LoadU(d, b + y * sb));
        acc = hn::MulAdd(diff, diff, acc);
    }
    return HSum8(acc);
#elif HWY_MAX_BYTES >= 16
    // Explicit low/high 4-lane halves: the same tree as the 8-lane path
    // (per-lane FMA chain over rows, then low+high, then ReduceSum). A scalar
    // loop here is reassociated freely under -ffast-math, and GCC 15 did so
    // differently in the two TUs that must agree.
    const hn::FixedTag<float, 4> d;
    auto lo = hn::Zero(d);
    auto hi = hn::Zero(d);
    for (int y = 0; y < 8; ++y) {
        const float* pa = a + y * sa;
        const float* pb = b + y * sb;
        const auto dlo = hn::Sub(hn::LoadU(d, pa), hn::LoadU(d, pb));
        const auto dhi = hn::Sub(hn::LoadU(d, pa + 4), hn::LoadU(d, pb + 4));
        lo = hn::MulAdd(dlo, dlo, lo);
        hi = hn::MulAdd(dhi, dhi, hi);
    }
    return hn::ReduceSum(d, hn::Add(lo, hi));
#else
    float acc = 0.f;
    for (int y = 0; y < 8; ++y) {
        const float* pa = a + y * sa;
        const float* pb = b + y * sb;
        for (int x = 0; x < 8; ++x) {
            const float t = pa[x] - pb[x];
            acc += t * t;
        }
    }
    return acc;
#endif
}
