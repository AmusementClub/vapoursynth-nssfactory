#include "cpu/bm/dct16.hpp"
#include "cpu/hwy_config.hpp"
#ifdef NSS_BM_KERNEL_LAB
#include "cpu/bm/kernel_lab.hpp"
#endif

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/bm/dct16.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

#include "cpu/bm/dct_codelet_adapter.hpp"
#if HWY_MAX_BYTES >= 64
#include "cpu/bm/dct_codelet_fwd_n16_is16.hpp"
#include "cpu/bm/dct_codelet_inv_n16_is16.hpp"
#ifdef NSS_BM_KERNEL_LAB
// Reuse the generated arithmetic verbatim, changing only its output sink.
// The offset in every generated ST is a compile-time row index.
#undef ST
#define ST(ptr, value, vstride, alignment) \
    hn::ScatterIndex((value), d, O + ((ptr) - O) / 16, \
                     hn::Mul(hn::Iota(hn::Rebind<int, D>(), 0), hn::Set(hn::Rebind<int, D>(), 16)))
#define nss_dct16_fwd_is16 nss_dct16_fwd_scatter
#define nss_dct16_inv_is16 nss_dct16_inv_scatter
#include "cpu/bm/dct_codelet_fwd_n16_is16.hpp"
#include "cpu/bm/dct_codelet_inv_n16_is16.hpp"
#undef nss_dct16_fwd_is16
#undef nss_dct16_inv_is16
#undef ST
#define ST(ptr, value, vstride, alignment) values[((ptr) - O) / 16] = (value)
#define nss_dct16_fwd_is16(dp, ip, op) nss_dct16_fwd_vectors(dp, ip, op, hn::Vec<D>* values)
#define nss_dct16_inv_is16(dp, ip, op) nss_dct16_inv_vectors(dp, ip, op, hn::Vec<D>* values)
#include "cpu/bm/dct_codelet_fwd_n16_is16.hpp"
#include "cpu/bm/dct_codelet_inv_n16_is16.hpp"
#undef nss_dct16_fwd_is16
#undef nss_dct16_inv_is16
#endif  // NSS_BM_KERNEL_LAB
// The is8 arithmetic is unchanged. Map its packed stride-8 loads to the
// actual stride-16 input and retain each result vector for the output sink.
#undef LD
#define LD(ptr, vstride, alignment) hn::LoadU(d, I + ((ptr) - I) * 2)
#undef ST
#define ST(ptr, value, vstride, alignment) values[((ptr) - O) / 8] = (value)
#define nss_dct16_fwd_is8(dp, ip, op) nss_dct16_fwd_vectors8(dp, ip, op, hn::Vec<D>* values)
#define nss_dct16_inv_is8(dp, ip, op) nss_dct16_inv_vectors8(dp, ip, op, hn::Vec<D>* values)
#include "cpu/bm/dct_codelet_fwd_n16_is8.hpp"
#include "cpu/bm/dct_codelet_inv_n16_is8.hpp"
#undef nss_dct16_fwd_is8
#undef nss_dct16_inv_is8
#endif
#undef VLEAVE
#undef VNEG
#undef VFNMS
#undef VFMS
#undef VFMA
#undef VMUL
#undef VSUB
#undef VADD
#undef ST
#undef LD
#undef LDK
#undef DVK
#undef V
#undef R

#if HWY_MAX_BYTES >= 64
using D8 = hn::FixedTag<float, 8>;
using V8 = hn::Vec<D8>;

static void Transpose8x8(D8 d8, const V8 r[8], V8 c[8]) {
    const auto t0 = hn::InterleaveLower(d8, r[0], r[1]);
    const auto t1 = hn::InterleaveUpper(d8, r[0], r[1]);
    const auto t2 = hn::InterleaveLower(d8, r[2], r[3]);
    const auto t3 = hn::InterleaveUpper(d8, r[2], r[3]);
    const auto t4 = hn::InterleaveLower(d8, r[4], r[5]);
    const auto t5 = hn::InterleaveUpper(d8, r[4], r[5]);
    const auto t6 = hn::InterleaveLower(d8, r[6], r[7]);
    const auto t7 = hn::InterleaveUpper(d8, r[6], r[7]);
    const hn::Repartition<double, D8> dd;
    const auto u0 = hn::BitCast(d8, hn::InterleaveLower(dd, hn::BitCast(dd, t0), hn::BitCast(dd, t2)));
    const auto u1 = hn::BitCast(d8, hn::InterleaveUpper(dd, hn::BitCast(dd, t0), hn::BitCast(dd, t2)));
    const auto u2 = hn::BitCast(d8, hn::InterleaveLower(dd, hn::BitCast(dd, t1), hn::BitCast(dd, t3)));
    const auto u3 = hn::BitCast(d8, hn::InterleaveUpper(dd, hn::BitCast(dd, t1), hn::BitCast(dd, t3)));
    const auto u4 = hn::BitCast(d8, hn::InterleaveLower(dd, hn::BitCast(dd, t4), hn::BitCast(dd, t6)));
    const auto u5 = hn::BitCast(d8, hn::InterleaveUpper(dd, hn::BitCast(dd, t4), hn::BitCast(dd, t6)));
    const auto u6 = hn::BitCast(d8, hn::InterleaveLower(dd, hn::BitCast(dd, t5), hn::BitCast(dd, t7)));
    const auto u7 = hn::BitCast(d8, hn::InterleaveUpper(dd, hn::BitCast(dd, t5), hn::BitCast(dd, t7)));
    c[0] = hn::ConcatLowerLower(d8, u4, u0);
    c[1] = hn::ConcatLowerLower(d8, u5, u1);
    c[2] = hn::ConcatLowerLower(d8, u6, u2);
    c[3] = hn::ConcatLowerLower(d8, u7, u3);
    c[4] = hn::ConcatUpperUpper(d8, u4, u0);
    c[5] = hn::ConcatUpperUpper(d8, u5, u1);
    c[6] = hn::ConcatUpperUpper(d8, u6, u2);
    c[7] = hn::ConcatUpperUpper(d8, u7, u3);
}

#include "cpu/bm/transpose8-inl.hpp"

#ifdef NSS_BM_KERNEL_LAB
static void Dct16Rows(const float* input, float* output, bool inverse) {
    const D8 d8;
    const hn::FixedTag<float, 16> d16;
    V8 r00[8], r01[8], r10[8], r11[8];
    V8 t00[8], t01[8], t10[8], t11[8];
    HWY_ALIGN float x[16 * 16];
    HWY_ALIGN float y[16 * 16];
    for (int row = 0; row < 8; ++row) {
        r00[row] = hn::LoadU(d8, input + row * 16);
        r01[row] = hn::LoadU(d8, input + row * 16 + 8);
        r10[row] = hn::LoadU(d8, input + (row + 8) * 16);
        r11[row] = hn::LoadU(d8, input + (row + 8) * 16 + 8);
    }
    Transpose8x8(d8, r00, t00);
    Transpose8x8(d8, r01, t01);
    Transpose8x8(d8, r10, t10);
    Transpose8x8(d8, r11, t11);
    for (int i = 0; i < 8; ++i) {
        hn::StoreU(hn::Combine(d16, t10[i], t00[i]), d16, x + i * 16);
        hn::StoreU(hn::Combine(d16, t11[i], t01[i]), d16, x + (i + 8) * 16);
    }
    inverse ? nss_dct16_inv_is16(d16, x, y) : nss_dct16_fwd_is16(d16, x, y);
    for (int i = 0; i < 8; ++i) {
        const auto y0 = hn::LoadU(d16, y + i * 16);
        const auto y1 = hn::LoadU(d16, y + (i + 8) * 16);
        t00[i] = hn::LowerHalf(d8, y0);
        t10[i] = hn::UpperHalf(d8, y0);
        t01[i] = hn::LowerHalf(d8, y1);
        t11[i] = hn::UpperHalf(d8, y1);
    }
    Transpose8x8(d8, t00, r00);
    Transpose8x8(d8, t01, r01);
    Transpose8x8(d8, t10, r10);
    Transpose8x8(d8, t11, r11);
    for (int row = 0; row < 8; ++row) {
        hn::StoreU(r00[row], d8, output + row * 16);
        hn::StoreU(r01[row], d8, output + row * 16 + 8);
        hn::StoreU(r10[row], d8, output + (row + 8) * 16);
        hn::StoreU(r11[row], d8, output + (row + 8) * 16 + 8);
    }
}
#endif
#endif

#ifdef NSS_BM_KERNEL_LAB
bool Dct16BaselineBatch(float* patches, int count, bool inverse) {
#if HWY_MAX_BYTES >= 64
    if (!patches || count < 1) {
        return false;
    }
    for (int patch = 0; patch < count; ++patch) {
        float* base = patches + patch * 16 * 16;
        const hn::FixedTag<float, 16> d16;
        HWY_ALIGN float columns[16 * 16];
        inverse ? nss_dct16_inv_is16(d16, base, columns) : nss_dct16_fwd_is16(d16, base, columns);
        Dct16Rows(columns, base, inverse);
    }
    return true;
#else
    (void)patches;
    (void)count;
    (void)inverse;
    return false;
#endif
}
#endif

#ifdef NSS_BM_KERNEL_LAB
#if HWY_MAX_BYTES >= 64
HWY_NOINLINE bool Dct16ScatterBatch(float* patches, int count, bool inverse) {
    if (!patches || count < 1) return false;
    const hn::FixedTag<float, 16> d;
    HWY_ALIGN float transposed[256];
    for (int p = 0; p < count; ++p) {
        float* base = patches + p * 256;
        if (inverse) {
            nss_dct16_inv_scatter(d, base, transposed);
            nss_dct16_inv_scatter(d, transposed, base);
        } else {
            nss_dct16_fwd_scatter(d, base, transposed);
            nss_dct16_fwd_scatter(d, transposed, base);
        }
    }
    return true;
}

static HWY_INLINE void Dct16VectorOutput(const float* input, float* output, bool inverse) {
    const hn::FixedTag<float, 16> d;
    const D8 d8;
    hn::Vec<decltype(d)> values[16];
    if (inverse) nss_dct16_inv_vectors(d, input, output, values);
    else nss_dct16_fwd_vectors(d, input, output, values);
    // Emit one quadrant at a time. The sink keeps the codelet's result vectors
    // available without a row-major y[256] store followed by a reload.
    for (int half = 0; half < 2; ++half) {
        V8 rows[8], columns[8];
        for (int i = 0; i < 8; ++i) rows[i] = hn::LowerHalf(d8, values[half * 8 + i]);
        Transpose8x8(d8, rows, columns);
        for (int i = 0; i < 8; ++i) hn::StoreU(columns[i], d8, output + i * 16 + half * 8);
        for (int i = 0; i < 8; ++i) rows[i] = hn::UpperHalf(d8, values[half * 8 + i]);
        Transpose8x8(d8, rows, columns);
        for (int i = 0; i < 8; ++i) hn::StoreU(columns[i], d8, output + (i + 8) * 16 + half * 8);
    }
}

HWY_NOINLINE bool Dct16VectorsBatch(float* patches, int count, bool inverse) {
    if (!patches || count < 1) return false;
    HWY_ALIGN float transposed[256];
    for (int p = 0; p < count; ++p) {
        float* base = patches + p * 256;
        Dct16VectorOutput(base, transposed, inverse);
        Dct16VectorOutput(transposed, base, inverse);
    }
    return true;
}
#endif  // HWY_MAX_BYTES >= 64
#endif  // NSS_BM_KERNEL_LAB

#if HWY_MAX_BYTES >= 64
template <bool InlineTranspose>
static HWY_INLINE void Dct16VectorOutput8(const float* input, float* output, bool inverse) {
    const D8 d8;
    for (int half = 0; half < 2; ++half) {
        V8 values[16], columns[8];
        if (inverse) nss_dct16_inv_vectors8(d8, input + half * 8, output, values);
        else nss_dct16_fwd_vectors8(d8, input + half * 8, output, values);
        if constexpr (InlineTranspose) Transpose8x8Inline(d8, values, columns);
        else Transpose8x8(d8, values, columns);
        for (int i = 0; i < 8; ++i) hn::StoreU(columns[i], d8, output + (half * 8 + i) * 16);
        if constexpr (InlineTranspose) Transpose8x8Inline(d8, values + 8, columns);
        else Transpose8x8(d8, values + 8, columns);
        for (int i = 0; i < 8; ++i) hn::StoreU(columns[i], d8, output + (half * 8 + i) * 16 + 8);
    }
}

#ifdef NSS_BM_KERNEL_LAB
HWY_NOINLINE bool Dct16Vectors8Batch(float* patches, int count, bool inverse) {
    if (!patches || count < 1) return false;
    HWY_ALIGN float transposed[256];
    for (int p = 0; p < count; ++p) {
        float* base = patches + p * 256;
        Dct16VectorOutput8<false>(base, transposed, inverse);
        Dct16VectorOutput8<false>(transposed, base, inverse);
    }
    return true;
}
#endif

HWY_NOINLINE bool Dct16Inline8Batch(float* patches, int count, bool inverse) {
    if (!patches || count < 1) return false;
    HWY_ALIGN float transposed[256];
    for (int p = 0; p < count; ++p) {
        float* base = patches + p * 256;
        Dct16VectorOutput8<true>(base, transposed, inverse);
        Dct16VectorOutput8<true>(transposed, base, inverse);
    }
    return true;
}
#endif

bool Dct16Batch(float* patches, int count, bool inverse) {
#if HWY_MAX_BYTES >= 64
    return Dct16Inline8Batch(patches, count, inverse);
#else
    (void)patches;
    (void)count;
    (void)inverse;
    return false;
#endif
}

#ifdef NSS_BM_KERNEL_LAB
detail::Dct16Kernel Dct16LabKernel(int variant) {
#if HWY_MAX_BYTES >= 64
    switch (variant) {
        case 0: return &Dct16BaselineBatch;
        case 1: return &Dct16ScatterBatch;
        case 2: return &Dct16VectorsBatch;
        case 3: return &Dct16Vectors8Batch;
        case 4: return &Dct16Inline8Batch;
    }
#endif
    return nullptr;
}
#endif

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss::detail {
HWY_EXPORT(Dct16Batch);
#ifdef NSS_BM_KERNEL_LAB
HWY_EXPORT(Dct16LabKernel);
Dct16Kernel dct16_lab_kernel(int variant) {
    return HWY_DYNAMIC_DISPATCH(Dct16LabKernel)(variant);
}
#endif

bool dct16_2d_batch_fast(float* patches, int count, bool inverse) {
    return HWY_DYNAMIC_DISPATCH(Dct16Batch)(patches, count, inverse);
}

}  // namespace nss::detail
#endif
