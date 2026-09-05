#include "cpu/bm/dct12.hpp"
#include "cpu/hwy_config.hpp"

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

bool Dct12Batch(float* patches, int count, bool inverse) {
#if HWY_MAX_BYTES >= 32
    if (!patches || count < 1) {
        return false;
    }
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

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss::detail {
HWY_EXPORT(Dct12Batch);

bool dct12_2d_batch_fast(float* patches, int count, bool inverse) {
    return HWY_DYNAMIC_DISPATCH(Dct12Batch)(patches, count, inverse);
}

}  // namespace nss::detail
#endif
