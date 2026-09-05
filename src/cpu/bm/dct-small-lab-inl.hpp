// Target-local experiments. Matrix/orthonormal arithmetic remains distinct
// from the FFTW-scaled b8/g8 filter. Baseline processes all rows before columns.
#if HWY_MAX_BYTES >= 32
#include "cpu/bm/transpose8-inl.hpp"
static HWY_INLINE void Dct8Rows8Inline(float* base, bool inverse) {
    const D8 d8;
    V8 r[8];
    V8 t[8];
    V8 y[8];
    for (int i = 0; i < 8; ++i) {
        r[i] = hn::LoadU(d8, base + i * 8);
    }
    Transpose8x8Inline(d8, r, t);
    Dct8Apply(d8, t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], inverse, y);
    Transpose8x8Inline(d8, y, r);
    for (int i = 0; i < 8; ++i) {
        hn::StoreU(r[i], d8, base + i * 8);
    }
}

#if HWY_MAX_BYTES >= 64
static HWY_INLINE void Dct8Rows16Inline(float* base, bool inverse) {
    const D8 d8;
    const hn::FixedTag<float, 16> d16;
    V8 r0[8];
    V8 r1[8];
    V8 a[8];
    V8 b[8];
    for (int i = 0; i < 8; ++i) {
        r0[i] = hn::LoadU(d8, base + i * 8);
        r1[i] = hn::LoadU(d8, base + (8 + i) * 8);
    }
    Transpose8x8Inline(d8, r0, a);
    Transpose8x8Inline(d8, r1, b);
    hn::Vec<decltype(d16)> in16[8];
    hn::Vec<decltype(d16)> out16[8];
    for (int i = 0; i < 8; ++i) {
        in16[i] = hn::Combine(d16, b[i], a[i]);
    }
    Dct8Apply(d16, in16[0], in16[1], in16[2], in16[3], in16[4], in16[5], in16[6], in16[7], inverse, out16);
    for (int i = 0; i < 8; ++i) {
        a[i] = hn::LowerHalf(d8, out16[i]);
        b[i] = hn::UpperHalf(d8, out16[i]);
    }
    Transpose8x8Inline(d8, a, r0);
    Transpose8x8Inline(d8, b, r1);
    for (int i = 0; i < 8; ++i) {
        hn::StoreU(r0[i], d8, base + i * 8);
        hn::StoreU(r1[i], d8, base + (8 + i) * 8);
    }
}

#endif

#if HWY_MAX_BYTES >= 64
static HWY_INLINE void Dct8PairInReg(float* base, bool inverse) {
    const D8 d8;
    V8 r[8], a[8], b[8];
    for (int i = 0; i < 8; ++i) r[i] = hn::LoadU(d8, base + i * 8);
    Transpose8x8Inline(d8, r, a);
    const hn::FixedTag<float, 16> d16;
    hn::Vec<decltype(d16)> x[8], y[8];
    for (int i = 0; i < 8; ++i) r[i] = hn::LoadU(d8, base + 64 + i * 8);
    Transpose8x8Inline(d8, r, b);
    for (int i = 0; i < 8; ++i) x[i] = hn::Combine(d16, b[i], a[i]);
    Dct8Apply(d16, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], inverse, y);
    for (int i = 0; i < 8; ++i) {
        a[i] = hn::LowerHalf(d8, y[i]);
        b[i] = hn::UpperHalf(d8, y[i]);
    }
    Transpose8x8Inline(d8, a, r);
    Transpose8x8Inline(d8, b, a);
    for (int i = 0; i < 8; ++i) x[i] = hn::Combine(d16, a[i], r[i]);
    Dct8Apply(d16, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], inverse, y);
    for (int i = 0; i < 8; ++i) {
        hn::StoreU(hn::LowerHalf(d8, y[i]), d8, base + i * 8);
        hn::StoreU(hn::UpperHalf(d8, y[i]), d8, base + 64 + i * 8);
    }
}
#endif

#endif

template <int Block, int Variant>
HWY_NOINLINE bool DctSmallBatch(float* base, int count, bool inverse) {
#if HWY_MAX_BYTES >= 32
    if (!base || count < 1) return false;
    if constexpr (Block == 8 && Variant == 2) {
        int p = 0;
#if HWY_MAX_BYTES >= 64
        for (; p + 1 < count; p += 2) Dct8PairInReg(base + p * 64, inverse);
#endif
        // Fusing the eight-lane tail changes rounding on the tested compiler.
        // Keep the original packed tail instead.
        for (; p < count; ++p) {
            DctLines(base + p * 64, 8, 8, 1, 8, inverse);
            DctLines(base + p * 64, 8, 1, 8, 8, inverse);
        }
        return true;
    }
    if constexpr (Block == 8 && Variant == 1) {
        int p = 0;
#if HWY_MAX_BYTES >= 64
        for (; p + 1 < count; p += 2) Dct8Rows16Inline(base + p * 64, inverse);
#endif
        for (; p < count; ++p) Dct8Rows8Inline(base + p * 64, inverse);
    } else {
        DctLines(base, Block, Block, 1, count * Block, inverse);
    }
    int p = 0;
#if HWY_MAX_BYTES >= 64
    if constexpr (Block == 8)
        for (; p + 1 < count; p += 2) Dct8Cols2Patches(base + p * 64, base + (p + 1) * 64, inverse);
#endif
    for (; p < count; ++p) {
        if constexpr (Block == 4) Dct4PatchCols(base + p * 16, inverse);
        else DctLines(base + p * 64, 8, 1, 8, 8, inverse);
    }
    return true;
#else
    (void)base; (void)count; (void)inverse;
    return false;
#endif
}

nss::detail::DctKernel ResolveDctSmallLab(int block, int variant) {
#if HWY_MAX_BYTES >= 32
    if (block == 4) {
        if (variant == 0) return DctSmallBatch<4, 0>;
    }
    if (block == 8) {
        switch (variant) {
            case 0: return DctSmallBatch<8, 0>;
            case 1: return DctSmallBatch<8, 1>;
            case 2: return DctSmallBatch<8, 2>;
        }
    }
#endif
    return nullptr;
}
