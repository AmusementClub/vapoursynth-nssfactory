#include "cpu/bm/dct12.hpp"
#include "cpu/hwy_config.hpp"
#ifdef NSS_BM_KERNEL_LAB
#include "cpu/bm/kernel_lab.hpp"
#endif

#include <algorithm>
#include <cstdint>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/bm/dct12.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

#include "cpu/bm/dct_codelet_adapter.hpp"
#if HWY_MAX_BYTES >= 32
#include "cpu/bm/dct_codelet_fwd_n12_is8.hpp"
#include "cpu/bm/dct_codelet_inv_n12_is8.hpp"
#endif
#if HWY_MAX_BYTES >= 64
#include "cpu/bm/dct_codelet_fwd_n12_is16.hpp"
#include "cpu/bm/dct_codelet_inv_n12_is16.hpp"
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

#if HWY_MAX_BYTES >= 32
#if HWY_MAX_BYTES >= 64
using D12 = hn::FixedTag<float, 16>;
static constexpr int kD12Lanes = 16;
#else
using D12 = hn::FixedTag<float, 8>;
static constexpr int kD12Lanes = 8;
#endif

static HWY_INLINE void Dct12Codelet(D12 d, const float* in, float* out, bool inverse) {
#if HWY_MAX_BYTES >= 64
    inverse ? nss_dct12_inv_is16(d, in, out) : nss_dct12_fwd_is16(d, in, out);
#else
    inverse ? nss_dct12_inv_is8(d, in, out) : nss_dct12_fwd_is8(d, in, out);
#endif
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
static void Dct12Rows(float* base, int patch_count, bool inverse) {
    const D12 d;
    const hn::RebindToSigned<D12> di;
    const auto indices = hn::Mul(hn::Iota(di, 0), hn::Set(di, 12));
    HWY_ALIGN float x[12 * kD12Lanes];
    HWY_ALIGN float y[12 * kD12Lanes];
    const int total_rows = patch_count * 12;
    for (int row0 = 0; row0 < total_rows; row0 += kD12Lanes) {
        const std::size_t count = static_cast<std::size_t>(std::min(kD12Lanes, total_rows - row0));
        float* rows = base + static_cast<std::size_t>(row0) * 12;
        for (int sample = 0; sample < 12; ++sample) {
            hn::StoreU(hn::GatherIndexN(d, rows + sample, indices, count), d, x + sample * kD12Lanes);
        }
        Dct12Codelet(d, x, y, inverse);
        for (int sample = 0; sample < 12; ++sample) {
            hn::ScatterIndexN(hn::LoadU(d, y + sample * kD12Lanes), d, rows + sample, indices, count);
        }
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
static void Dct12Cols(float* base, bool inverse) {
    const D12 d;
    HWY_ALIGN float x[12 * kD12Lanes];
    HWY_ALIGN float y[12 * kD12Lanes];
    for (int col0 = 0; col0 < 12; col0 += kD12Lanes) {
        const std::size_t count = static_cast<std::size_t>(std::min(kD12Lanes, 12 - col0));
        for (int row = 0; row < 12; ++row) {
            hn::StoreU(hn::LoadN(d, base + row * 12 + col0, count), d, x + row * kD12Lanes);
        }
        Dct12Codelet(d, x, y, inverse);
        for (int row = 0; row < 12; ++row) {
            hn::StoreN(hn::LoadU(d, y + row * kD12Lanes), d, base + row * 12 + col0, count);
        }
    }
}
#endif

#ifdef NSS_BM_KERNEL_LAB
#include "cpu/bm/dct12-lab-inl.hpp"
#endif

bool Dct12Batch(float* patches, int count, bool inverse) {
#if HWY_MAX_BYTES >= 32
    if (!patches || count < 1) {
        return false;
    }
#if defined(NSS_BM_KERNEL_LAB) && defined(NSS_LAB_DCT12_VARIANT)
    return Dct12LayoutBatch<NSS_LAB_DCT12_VARIANT>(patches, count, inverse);
#endif
    Dct12Rows(patches, count, inverse);
    for (int patch = 0; patch < count; ++patch) {
        Dct12Cols(patches + static_cast<std::size_t>(patch) * 12 * 12, inverse);
    }
    return true;
#else
    (void)patches;
    (void)count;
    (void)inverse;
    return false;
#endif
}

#ifdef NSS_BM_KERNEL_LAB
// Variant zero remains the original route even in a selected candidate build.
static bool Dct12OriginalBatch(float* patches, int count, bool inverse) {
#if HWY_MAX_BYTES >= 32
    if (!patches || count < 1) return false;
    Dct12Rows(patches, count, inverse);
    for (int p = 0; p < count; ++p) Dct12Cols(patches + p * 144, inverse);
    return true;
#else
    (void)patches; (void)count; (void)inverse;
    return false;
#endif
}
nss::detail::DctKernel ResolveDct12Lab(int variant) {
#if HWY_MAX_BYTES >= 32
    switch (variant) {
        case 0: return Dct12OriginalBatch;
        case 1: return Dct12LayoutBatch<1>;
        case 2: return Dct12LayoutBatch<2>;
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
HWY_EXPORT(Dct12Batch);
#ifdef NSS_BM_KERNEL_LAB
HWY_EXPORT(ResolveDct12Lab);
DctKernel dct12_lab_kernel(int variant) { return HWY_DYNAMIC_DISPATCH(ResolveDct12Lab)(variant); }
#endif

bool dct12_2d_batch_fast(float* patches, int count, bool inverse) {
    return HWY_DYNAMIC_DISPATCH(Dct12Batch)(patches, count, inverse);
}

}  // namespace nss::detail
#endif
