// Re-included per Highway target. Layout experiments only; generated arithmetic
// and row-then-column ordering match Dct12Batch. No overread of partial tiles.
#if HWY_MAX_BYTES >= 32
using D8 = hn::FixedTag<float, 8>;
using V8 = hn::Vec<D8>;
#include "cpu/bm/transpose8-inl.hpp"
#include "cpu/bm/dct_codelet_adapter.hpp"
#undef LD
#undef ST
#define LD(ptr, vs, a) input[((ptr) - I) / 8]
#define ST(ptr, value, vs, a) output[((ptr) - O) / 8] = (value)
#define nss_dct12_fwd_is8(dp, ip, op) Dct12FwdVectors8(dp, ip, op, const hn::Vec<D>* input, hn::Vec<D>* output)
#define nss_dct12_inv_is8(dp, ip, op) Dct12InvVectors8(dp, ip, op, const hn::Vec<D>* input, hn::Vec<D>* output)
#include "cpu/bm/dct_codelet_fwd_n12_is8.hpp"
#include "cpu/bm/dct_codelet_inv_n12_is8.hpp"
#undef nss_dct12_fwd_is8
#undef nss_dct12_inv_is8
// Sink variant: emit each frequency vector directly into transposed rows.
#undef ST
#define ST(ptr, value, vs, a) hn::StoreN((value), d, O + (((ptr) - O) / 8) * 12, rows)
#define nss_dct12_fwd_is8(dp, ip, op) Dct12FwdSink8(dp, ip, op, const hn::Vec<D>* input, std::size_t rows)
#define nss_dct12_inv_is8(dp, ip, op) Dct12InvSink8(dp, ip, op, const hn::Vec<D>* input, std::size_t rows)
#include "cpu/bm/dct_codelet_fwd_n12_is8.hpp"
#include "cpu/bm/dct_codelet_inv_n12_is8.hpp"
#undef nss_dct12_fwd_is8
#undef nss_dct12_inv_is8
#if HWY_MAX_BYTES >= 64
#undef LD
#undef ST
#define LD(ptr, vs, a) input[((ptr) - I) / 16]
#define ST(ptr, value, vs, a) output[((ptr) - O) / 16] = (value)
#define nss_dct12_fwd_is16(dp, ip, op) Dct12FwdVectors16(dp, ip, op, const hn::Vec<D>* input, hn::Vec<D>* output)
#define nss_dct12_inv_is16(dp, ip, op) Dct12InvVectors16(dp, ip, op, const hn::Vec<D>* input, hn::Vec<D>* output)
#include "cpu/bm/dct_codelet_fwd_n12_is16.hpp"
#include "cpu/bm/dct_codelet_inv_n12_is16.hpp"
#undef nss_dct12_fwd_is16
#undef nss_dct12_inv_is16
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

// Each lane is an independent row; output is [frequency][original row].
template <int Rows, int Cols>
static HWY_INLINE void Dct12LoadTile(const float* src, V8* input) {
    const D8 d;
    V8 r[8], c[8];
    for (int i = 0; i < 8; ++i)
        r[i] = i < Rows ? hn::LoadN(d, src + i * 12, Cols) : hn::Zero(d);
    Transpose8x8Inline(d, r, c);
    for (int i = 0; i < Cols; ++i) input[i] = c[i];
}

template <int Rows>
static HWY_INLINE void Dct12TileRows(const float* src, float* dst, bool inverse) {
    const D8 d;
    V8 input[12];
    Dct12LoadTile<Rows, 8>(src, input);
    Dct12LoadTile<Rows, 4>(src + 8, input + 8);
    // I supplies generated constant address offsets only; no input is read
    // through this array. Its extent keeps every address token in bounds.
    float offsets[96];
    inverse ? Dct12InvSink8(d, offsets, dst, input, Rows)
            : Dct12FwdSink8(d, offsets, dst, input, Rows);
}

static HWY_INLINE void Dct12TilePass(const float* src, float* dst, bool inverse) {
    Dct12TileRows<8>(src, dst, inverse);
    Dct12TileRows<4>(src + 96, dst + 8, inverse);
}

static HWY_INLINE void Dct12CrossPass(const float* src, float* dst, int patches, bool inverse) {
    const D12 d;
    float offsets[192];  // Address tokens only; generated offsets fit this object.
    const hn::RebindToSigned<D12> di;
    const auto index = hn::Mul(hn::Iota(di, 0), hn::Set(di, 12));
    for (int row0 = 0; row0 < patches * 12; row0 += kD12Lanes) {
        const int rows = std::min(kD12Lanes, patches * 12 - row0);
        hn::Vec<D12> input[12], output[12];
        for (int i = 0; i < 12; ++i) input[i] = hn::GatherIndexN(d, src + row0 * 12 + i, index, rows);
        // Dummy pointers remain inside a sufficiently large real array; adapters
        // use their generated offsets only as vector-array indices.
#if HWY_MAX_BYTES >= 64
        inverse ? Dct12InvVectors16(d, offsets, offsets, input, output)
                : Dct12FwdVectors16(d, offsets, offsets, input, output);
#else
        inverse ? Dct12InvVectors8(d, offsets, offsets, input, output)
                : Dct12FwdVectors8(d, offsets, offsets, input, output);
#endif
        for (int consumed = 0; consumed < rows;) {
            const int row = row0 + consumed;
            const int n = std::min(rows - consumed, 12 - row % 12);
            const auto lanes = hn::IndicesFromVec(d, hn::Add(hn::Iota(di, 0), hn::Set(di, consumed)));
            for (int i = 0; i < 12; ++i)
                hn::StoreN(hn::TableLookupLanes(output[i], lanes), d,
                           dst + (row / 12) * 144 + i * 12 + row % 12, n);
            consumed += n;
        }
    }
}

template <int Variant>
HWY_NOINLINE bool Dct12LayoutBatch(float* patches, int count, bool inverse) {
    if (!patches || count < 1) return false;
    // Four 12-row patches fill three complete 16-row batches without a heap allocation.
    HWY_ALIGN float tmp[4 * 144];
    if constexpr (Variant == 1) {
        for (int p = 0; p < count; p += 4) {
            const int n = std::min(4, count - p);
            Dct12CrossPass(patches + p * 144, tmp, n, inverse);
            Dct12CrossPass(tmp, patches + p * 144, n, inverse);
        }
    } else {
        for (int p = 0; p < count; ++p) {
            Dct12TilePass(patches + p * 144, tmp, inverse);
            Dct12TilePass(tmp, patches + p * 144, inverse);
        }
    }
    return true;
}
#endif
