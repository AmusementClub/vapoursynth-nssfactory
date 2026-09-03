#include "nss/cpu_api.hpp"
#include "cpu/hwy_config.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/bm/dct8.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

#include "cpu/bm/dct_codelet_adapter.hpp"
#if HWY_MAX_BYTES >= 32
#include "cpu/bm/dct_codelet_fwd_n16_is8.hpp"
#include "cpu/bm/dct_codelet_inv_n16_is8.hpp"
#include "cpu/bm/dct_codelet_fwd_n32_is8.hpp"
#include "cpu/bm/dct_codelet_inv_n32_is8.hpp"
#include "cpu/bm/dct_codelet_fwd_n64_is8.hpp"
#include "cpu/bm/dct_codelet_inv_n64_is8.hpp"
#endif
#if HWY_MAX_BYTES >= 64
#include "cpu/bm/dct_codelet_fwd_n16_is16.hpp"
#include "cpu/bm/dct_codelet_inv_n16_is16.hpp"
#include "cpu/bm/dct_codelet_fwd_n32_is16.hpp"
#include "cpu/bm/dct_codelet_inv_n32_is16.hpp"
#include "cpu/bm/dct_codelet_fwd_n64_is16.hpp"
#include "cpu/bm/dct_codelet_inv_n64_is16.hpp"
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

// Orthonormal DCT-II matrix, row k = frequency, column i = sample.
template <int N>
const float* DctTable() {
    static float m[static_cast<size_t>(N) * static_cast<size_t>(N)];
    static std::once_flag once;
    std::call_once(once, [] {
        const double s0 = std::sqrt(1.0 / static_cast<double>(N));
        const double s = std::sqrt(2.0 / static_cast<double>(N));
        constexpr double pi = 3.14159265358979323846264338327950288;
        for (int k = 0; k < N; ++k) {
            for (int i = 0; i < N; ++i) {
                const double c = ((k == 0) ? s0 : s) *
                                 std::cos(pi * (static_cast<double>(i) + 0.5) * static_cast<double>(k) /
                                          static_cast<double>(N));
                m[static_cast<size_t>(k) * static_cast<size_t>(N) + static_cast<size_t>(i)] =
                    static_cast<float>(c);
            }
        }
    });
    return m;
}

static const float* DctMatrix(int n) {
    switch (n) {
        case 2:
            return DctTable<2>();
        case 4:
            return DctTable<4>();
        case 8:
            return DctTable<8>();
        case 16:
            return DctTable<16>();
        case 32:
            return DctTable<32>();
        case 64:
            return DctTable<64>();
        default:
            return nullptr;
    }
}

#if HWY_MAX_BYTES >= 32
// Lanes(d) independent 8-point orthonormal DCT-II / IDCT-III on registers.
template <class D>
HWY_INLINE void Dct8Apply(D d, hn::Vec<D> x0, hn::Vec<D> x1, hn::Vec<D> x2, hn::Vec<D> x3, hn::Vec<D> x4,
                          hn::Vec<D> x5, hn::Vec<D> x6, hn::Vec<D> x7, bool inverse, hn::Vec<D>* y) {
    const auto s0 = hn::Set(d, 0.3535533905932738f);
    const auto s1 = hn::Set(d, 0.4903926402016152f);
    const auto s2 = hn::Set(d, 0.4619397662556434f);
    const auto s3 = hn::Set(d, 0.4157348061512726f);
    const auto s4 = hn::Set(d, 0.3535533905932738f);
    const auto s5 = hn::Set(d, 0.2777851165098011f);
    const auto s6 = hn::Set(d, 0.1913417161825449f);
    const auto s7 = hn::Set(d, 0.0975451610080642f);
    if (!inverse) {
        y[0] = hn::Mul(s0, hn::Add(hn::Add(hn::Add(x0, x1), hn::Add(x2, x3)), hn::Add(hn::Add(x4, x5), hn::Add(x6, x7))));
        y[1] = hn::MulAdd(s1, x0,
                          hn::MulAdd(s3, x1,
                                     hn::MulAdd(s5, x2,
                                                hn::MulAdd(s7, x3,
                                                           hn::MulAdd(hn::Neg(s7), x4,
                                                                      hn::MulAdd(hn::Neg(s5), x5,
                                                                                 hn::MulAdd(hn::Neg(s3), x6, hn::Mul(hn::Neg(s1), x7))))))));
        y[2] = hn::MulAdd(s2, x0,
                          hn::MulAdd(s6, x1,
                                     hn::MulAdd(hn::Neg(s6), x2,
                                                hn::MulAdd(hn::Neg(s2), x3,
                                                           hn::MulAdd(hn::Neg(s2), x4,
                                                                      hn::MulAdd(hn::Neg(s6), x5,
                                                                                 hn::MulAdd(s6, x6, hn::Mul(s2, x7))))))));
        y[3] = hn::MulAdd(s3, x0,
                          hn::MulAdd(hn::Neg(s7), x1,
                                     hn::MulAdd(hn::Neg(s1), x2,
                                                hn::MulAdd(hn::Neg(s5), x3,
                                                           hn::MulAdd(s5, x4,
                                                                      hn::MulAdd(s1, x5, hn::MulAdd(s7, x6, hn::Mul(hn::Neg(s3), x7))))))));
        y[4] = hn::MulAdd(s4, x0,
                          hn::MulAdd(hn::Neg(s4), x1,
                                     hn::MulAdd(hn::Neg(s4), x2,
                                                hn::MulAdd(s4, x3,
                                                           hn::MulAdd(s4, x4,
                                                                      hn::MulAdd(hn::Neg(s4), x5,
                                                                                 hn::MulAdd(hn::Neg(s4), x6, hn::Mul(s4, x7))))))));
        y[5] = hn::MulAdd(s5, x0,
                          hn::MulAdd(hn::Neg(s1), x1,
                                     hn::MulAdd(s7, x2,
                                                hn::MulAdd(s3, x3,
                                                           hn::MulAdd(hn::Neg(s3), x4,
                                                                      hn::MulAdd(hn::Neg(s7), x5,
                                                                                 hn::MulAdd(s1, x6, hn::Mul(hn::Neg(s5), x7))))))));
        y[6] = hn::MulAdd(s6, x0,
                          hn::MulAdd(hn::Neg(s2), x1,
                                     hn::MulAdd(s2, x2,
                                                hn::MulAdd(hn::Neg(s6), x3,
                                                           hn::MulAdd(hn::Neg(s6), x4,
                                                                      hn::MulAdd(s2, x5, hn::MulAdd(hn::Neg(s2), x6, hn::Mul(s6, x7))))))));
        y[7] = hn::MulAdd(s7, x0,
                          hn::MulAdd(hn::Neg(s5), x1,
                                     hn::MulAdd(s3, x2,
                                                hn::MulAdd(hn::Neg(s1), x3,
                                                           hn::MulAdd(s1, x4,
                                                                      hn::MulAdd(hn::Neg(s3), x5,
                                                                                 hn::MulAdd(s5, x6, hn::Mul(hn::Neg(s7), x7))))))));
    } else {
        y[0] = hn::MulAdd(s0, x0,
                          hn::MulAdd(s1, x1,
                                     hn::MulAdd(s2, x2,
                                                hn::MulAdd(s3, x3, hn::MulAdd(s4, x4, hn::MulAdd(s5, x5, hn::MulAdd(s6, x6, hn::Mul(s7, x7))))))));
        y[1] = hn::MulAdd(s0, x0,
                          hn::MulAdd(s3, x1,
                                     hn::MulAdd(s6, x2,
                                                hn::MulAdd(hn::Neg(s7), x3,
                                                           hn::MulAdd(hn::Neg(s4), x4,
                                                                      hn::MulAdd(hn::Neg(s1), x5,
                                                                                 hn::MulAdd(hn::Neg(s2), x6, hn::Mul(hn::Neg(s5), x7))))))));
        y[2] = hn::MulAdd(s0, x0,
                          hn::MulAdd(s5, x1,
                                     hn::MulAdd(hn::Neg(s6), x2,
                                                hn::MulAdd(hn::Neg(s1), x3,
                                                           hn::MulAdd(hn::Neg(s4), x4,
                                                                      hn::MulAdd(s7, x5, hn::MulAdd(s2, x6, hn::Mul(s3, x7))))))));
        y[3] = hn::MulAdd(s0, x0,
                          hn::MulAdd(s7, x1,
                                     hn::MulAdd(hn::Neg(s2), x2,
                                                hn::MulAdd(hn::Neg(s5), x3,
                                                           hn::MulAdd(s4, x4, hn::MulAdd(s3, x5, hn::MulAdd(hn::Neg(s6), x6, hn::Mul(hn::Neg(s1), x7))))))));
        y[4] = hn::MulAdd(s0, x0,
                          hn::MulAdd(hn::Neg(s7), x1,
                                     hn::MulAdd(hn::Neg(s2), x2,
                                                hn::MulAdd(s5, x3,
                                                           hn::MulAdd(s4, x4,
                                                                      hn::MulAdd(hn::Neg(s3), x5,
                                                                                 hn::MulAdd(hn::Neg(s6), x6, hn::Mul(s1, x7))))))));
        y[5] = hn::MulAdd(s0, x0,
                          hn::MulAdd(hn::Neg(s5), x1,
                                     hn::MulAdd(hn::Neg(s6), x2,
                                                hn::MulAdd(s1, x3,
                                                           hn::MulAdd(hn::Neg(s4), x4,
                                                                      hn::MulAdd(hn::Neg(s7), x5,
                                                                                 hn::MulAdd(s2, x6, hn::Mul(hn::Neg(s3), x7))))))));
        y[6] = hn::MulAdd(s0, x0,
                          hn::MulAdd(hn::Neg(s3), x1,
                                     hn::MulAdd(s6, x2,
                                                hn::MulAdd(s7, x3,
                                                           hn::MulAdd(hn::Neg(s4), x4,
                                                                      hn::MulAdd(s1, x5, hn::MulAdd(hn::Neg(s2), x6, hn::Mul(s5, x7))))))));
        y[7] = hn::MulAdd(s0, x0,
                          hn::MulAdd(hn::Neg(s1), x1,
                                     hn::MulAdd(s2, x2,
                                                hn::MulAdd(hn::Neg(s3), x3,
                                                           hn::MulAdd(s4, x4,
                                                                      hn::MulAdd(hn::Neg(s5), x5, hn::MulAdd(s6, x6, hn::Mul(hn::Neg(s7), x7))))))));
    }
}

template <class D>
static void Dct8Packed(D d, float* x, float* y, bool inverse) {
    const int L = static_cast<int>(hn::Lanes(d));
    hn::Vec<D> out[8];
    Dct8Apply(d, hn::LoadU(d, x + 0 * L), hn::LoadU(d, x + 1 * L), hn::LoadU(d, x + 2 * L), hn::LoadU(d, x + 3 * L),
              hn::LoadU(d, x + 4 * L), hn::LoadU(d, x + 5 * L), hn::LoadU(d, x + 6 * L), hn::LoadU(d, x + 7 * L), inverse,
              out);
    for (int i = 0; i < 8; ++i) {
        hn::StoreU(out[i], d, y + i * L);
    }
}

using D8 = hn::FixedTag<float, 8>;
using V8 = hn::Vec<D8>;

static void Transpose8x8Vec(D8 d8, const V8 r[8], V8 c[8]);

static int clampi8(int x, int lo, int hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static void LoadPatch8Inside(const float* src, int stride, int x, int y, V8* rows) {
    const D8 d8;
    for (int r = 0; r < 8; ++r) {
        rows[r] = hn::LoadU(d8, src + (y + r) * stride + x);
    }
}

static void LoadPatch8(const float* src, int stride, int x, int y, int w, int h, V8* rows) {
    if (x >= 0 && y >= 0 && x + 8 <= w && y + 8 <= h) {
        LoadPatch8Inside(src, stride, x, y, rows);
        return;
    }
    const D8 d8;
    HWY_ALIGN float tmp[8];
    for (int r = 0; r < 8; ++r) {
        const int yy = clampi8(y + r, 0, h - 1);
        for (int c = 0; c < 8; ++c) {
            tmp[c] = src[yy * stride + clampi8(x + c, 0, w - 1)];
        }
        rows[r] = hn::LoadU(d8, tmp);
    }
}

static void AccPatch8Inside(float* num, float* den, int stride, int x, int y, const V8* rows, float value_weight,
                            float den_weight) {
    const D8 d8;
    const auto vvalue = hn::Set(d8, value_weight);
    const auto vden = hn::Set(d8, den_weight);
    for (int r = 0; r < 8; ++r) {
        float* np = num + (y + r) * stride + x;
        float* dp = den + (y + r) * stride + x;
        hn::StoreU(hn::MulAdd(vvalue, rows[r], hn::LoadU(d8, np)), d8, np);
        hn::StoreU(hn::Add(hn::LoadU(d8, dp), vden), d8, dp);
    }
}

static void AccPatch8(float* num, float* den, int stride, int x, int y, int w, int h, const V8* rows,
                      float value_weight, float den_weight) {
    if (x >= 0 && y >= 0 && x + 8 <= w && y + 8 <= h) {
        AccPatch8Inside(num, den, stride, x, y, rows, value_weight, den_weight);
        return;
    }
    const D8 d8;
    HWY_ALIGN float tmp[8];
    for (int r = 0; r < 8; ++r) {
        const int yy = y + r;
        if (yy < 0 || yy >= h) {
            continue;
        }
        hn::StoreU(rows[r], d8, tmp);
        for (int c = 0; c < 8; ++c) {
            const int xx = x + c;
            if (xx < 0 || xx >= w) {
                continue;
            }
            num[yy * stride + xx] += value_weight * tmp[c];
            den[yy * stride + xx] += den_weight;
        }
    }
}

static void Transpose8x8Vec(D8 d8, const V8 r[8], V8 c[8]) {
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

// 8 contiguous 8-float rows: in-register transpose, 8-wide DCT, transpose back.
static void Dct8Rows8(float* base, bool inverse) {
    const D8 d8;
    V8 r[8];
    V8 t[8];
    V8 y[8];
    for (int i = 0; i < 8; ++i) {
        r[i] = hn::LoadU(d8, base + i * 8);
    }
    Transpose8x8Vec(d8, r, t);
    Dct8Apply(d8, t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], inverse, y);
    Transpose8x8Vec(d8, y, r);
    for (int i = 0; i < 8; ++i) {
        hn::StoreU(r[i], d8, base + i * 8);
    }
}

#if HWY_MAX_BYTES >= 64
// Column DCT of two 8x8 patches: 16 independent length-8 DCTs, no transpose.
static void Dct8Cols2Patches(float* p0, float* p1, bool inverse) {
    const D8 d8;
    const hn::FixedTag<float, 16> d16;
    hn::Vec<decltype(d16)> in16[8];
    hn::Vec<decltype(d16)> out16[8];
    for (int i = 0; i < 8; ++i) {
        in16[i] = hn::Combine(d16, hn::LoadU(d8, p1 + i * 8), hn::LoadU(d8, p0 + i * 8));
    }
    Dct8Apply(d16, in16[0], in16[1], in16[2], in16[3], in16[4], in16[5], in16[6], in16[7], inverse, out16);
    for (int i = 0; i < 8; ++i) {
        hn::StoreU(hn::LowerHalf(d8, out16[i]), d8, p0 + i * 8);
        hn::StoreU(hn::UpperHalf(d8, out16[i]), d8, p1 + i * 8);
    }
}

// 16 contiguous 8-float rows -> 16 independent length-8 DCTs on ZMM.
static void Dct8Rows16(float* base, bool inverse) {
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
    Transpose8x8Vec(d8, r0, a);
    Transpose8x8Vec(d8, r1, b);
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
    Transpose8x8Vec(d8, a, r0);
    Transpose8x8Vec(d8, b, r1);
    for (int i = 0; i < 8; ++i) {
        hn::StoreU(r0[i], d8, base + i * 8);
        hn::StoreU(r1[i], d8, base + (8 + i) * 8);
    }
}
#endif
#endif

// Independent length-n vectors: sample i of line v is at
// base[v * line_stride + i * sample_stride]. inverse = orthonormal DCT-III.
static void DctLines(float* base, int n, int line_stride, int sample_stride, int count, bool inverse) {
    if (n <= 1 || count <= 0) {
        return;
    }
    const hn::CappedTag<float, 16> d;
    const int L = static_cast<int>(hn::Lanes(d));
    HWY_ALIGN float x[64 * 16];
    HWY_ALIGN float y[64 * 16];
    for (int v0 = 0; v0 < count; v0 += L) {
        const int lanes = count - v0 < L ? count - v0 : L;
#if HWY_MAX_BYTES >= 32
        if (n == 8 && sample_stride == 1 && line_stride == 8) {
#if HWY_MAX_BYTES >= 64
            if (lanes == 16 && L == 16) {
                Dct8Rows16(base + v0 * 8, inverse);
                continue;
            }
#endif
            if (lanes == 8) {
                Dct8Rows8(base + v0 * 8, inverse);
                continue;
            }
        }
        if (n == 8 && line_stride == 1 && lanes == 8) {
            const D8 d8;
            for (int i = 0; i < 8; ++i) {
                hn::StoreU(hn::LoadU(d8, base + v0 + i * sample_stride), d8, x + i * 8);
            }
            Dct8Packed(d8, x, y, inverse);
            for (int i = 0; i < 8; ++i) {
                hn::StoreU(hn::LoadU(d8, y + i * 8), d8, base + v0 + i * sample_stride);
            }
            continue;
        }
        if (n == 8 && line_stride == 1 && lanes == L) {
            for (int i = 0; i < 8; ++i) {
                hn::StoreU(hn::LoadU(d, base + v0 + i * sample_stride), d, x + i * L);
            }
            Dct8Packed(d, x, y, inverse);
            for (int i = 0; i < 8; ++i) {
                hn::StoreU(hn::LoadU(d, y + i * L), d, base + v0 + i * sample_stride);
            }
            continue;
        }
#endif
        for (int i = 0; i < n; ++i) {
            for (int lane = 0; lane < lanes; ++lane) {
                x[i * L + lane] = base[(v0 + lane) * line_stride + i * sample_stride];
            }
            for (int lane = lanes; lane < L; ++lane) {
                x[i * L + lane] = 0.f;
            }
        }
#if HWY_MAX_BYTES >= 32
        if (n == 8) {
            Dct8Packed(d, x, y, inverse);
        } else if (n == 16) {
#if HWY_MAX_BYTES >= 64
            if (L == 16) {
                inverse ? nss_dct16_inv_is16(d, x, y) : nss_dct16_fwd_is16(d, x, y);
            } else
#endif
            {
                inverse ? nss_dct16_inv_is8(d, x, y) : nss_dct16_fwd_is8(d, x, y);
            }
        } else if (n == 32) {
#if HWY_MAX_BYTES >= 64
            if (L == 16) {
                inverse ? nss_dct32_inv_is16(d, x, y) : nss_dct32_fwd_is16(d, x, y);
            } else
#endif
            {
                inverse ? nss_dct32_inv_is8(d, x, y) : nss_dct32_fwd_is8(d, x, y);
            }
        } else if (n == 64) {
#if HWY_MAX_BYTES >= 64
            if (L == 16) {
                inverse ? nss_dct64_inv_is16(d, x, y) : nss_dct64_fwd_is16(d, x, y);
            } else
#endif
            {
                inverse ? nss_dct64_inv_is8(d, x, y) : nss_dct64_fwd_is8(d, x, y);
            }
        } else
#endif
        {
            const float* M = DctMatrix(n);
            if (!M) {
                return;
            }
            for (int outb = 0; outb < n; ++outb) {
                auto acc = hn::Zero(d);
                for (int inb = 0; inb < n; ++inb) {
                    const float c = inverse ? M[inb * n + outb] : M[outb * n + inb];
                    acc = hn::MulAdd(hn::Set(d, c), hn::LoadU(d, x + inb * L), acc);
                }
                hn::StoreU(acc, d, y + outb * L);
            }
        }
        for (int outb = 0; outb < n; ++outb) {
            for (int lane = 0; lane < lanes; ++lane) {
                base[(v0 + lane) * line_stride + outb * sample_stride] = y[outb * L + lane];
            }
        }
    }
}

void Dct1d(const float* in, float* out, int n) {
    if (n <= 1) {
        if (n == 1 && out != in) {
            out[0] = in[0];
        }
        return;
    }
    float tmp[64];
    std::memcpy(tmp, in, static_cast<size_t>(n) * sizeof(float));
    DctLines(tmp, n, 0, 1, 1, false);
    std::memcpy(out, tmp, static_cast<size_t>(n) * sizeof(float));
}

void Idct1d(const float* in, float* out, int n) {
    if (n <= 1) {
        if (n == 1 && out != in) {
            out[0] = in[0];
        }
        return;
    }
    float tmp[64];
    std::memcpy(tmp, in, static_cast<size_t>(n) * sizeof(float));
    DctLines(tmp, n, 0, 1, 1, true);
    std::memcpy(out, tmp, static_cast<size_t>(n) * sizeof(float));
}

void Dct2d(float* block, int n) {
    if (n <= 1) {
        return;
    }
    DctLines(block, n, n, 1, n, false);
    DctLines(block, n, 1, n, n, false);
}

void Idct2d(float* block, int n) {
    if (n <= 1) {
        return;
    }
    DctLines(block, n, n, 1, n, true);
    DctLines(block, n, 1, n, n, true);
}

void Dct8_1d(const float* in, float* out) {
    Dct1d(in, out, 8);
}

void Idct8_1d(const float* in, float* out) {
    Idct1d(in, out, 8);
}

void Dct8_2d(float* block) {
    Dct2d(block, 8);
}

void Idct8_2d(float* block) {
    Idct2d(block, 8);
}

void TransformLines(float* base, int n, int line_stride, int sample_stride, int count, bool inverse) {
    DctLines(base, n, line_stride, sample_stride, count, inverse);
}

void Bm3dFilterGroup(float* patches, int lda, int group, int k, int block, float sigma, bool wiener,
                     const float* ref_patches, float* weight_out, float* work) {
    if (group < 1 || block < 1 || k < 1 || !work) {
        if (weight_out) {
            *weight_out = 1.f;
        }
        return;
    }
    const int kk = std::min(k, group);
    const int area = block * block;
    const bool inplace = (lda == area && kk == group);
    float* cube = inplace ? patches : work;
    float* refw = work + static_cast<size_t>(group) * static_cast<size_t>(area);
    if (!inplace) {
        for (int i = 0; i < kk; ++i) {
            std::memcpy(cube + static_cast<size_t>(i) * area, patches + i * lda,
                        static_cast<size_t>(area) * sizeof(float));
        }
        if (kk < group) {
            std::memset(cube + static_cast<size_t>(kk) * area, 0,
                        static_cast<size_t>(group - kk) * static_cast<size_t>(area) * sizeof(float));
        }
    }
    auto dct2 = [&](float* c, bool inverse) {
        DctLines(c, block, block, 1, group * block, inverse);
#if HWY_MAX_BYTES >= 64
        if (block == 8) {
            int g = 0;
            for (; g + 1 < group; g += 2) {
                Dct8Cols2Patches(c + static_cast<size_t>(g) * 64, c + static_cast<size_t>(g + 1) * 64, inverse);
            }
            if (g < group) {
                DctLines(c + static_cast<size_t>(g) * 64, 8, 1, 8, 8, inverse);
            }
            return;
        }
#endif
        for (int g = 0; g < group; ++g) {
            DctLines(c + static_cast<size_t>(g) * area, block, 1, block, block, inverse);
        }
    };
    dct2(cube, false);
    if (wiener && ref_patches) {
        for (int i = 0; i < kk; ++i) {
            std::memcpy(refw + static_cast<size_t>(i) * area, ref_patches + i * lda,
                        static_cast<size_t>(area) * sizeof(float));
        }
        if (kk < group) {
            std::memset(refw + static_cast<size_t>(kk) * area, 0,
                        static_cast<size_t>(group - kk) * static_cast<size_t>(area) * sizeof(float));
        }
        dct2(refw, false);
    }

    DctLines(cube, group, 1, area, area, false);
    int kept = 0;
    float w2sum = 0.f;
    if (wiener && ref_patches) {
        DctLines(refw, group, 1, area, area, false);
        const float sig2 = sigma * sigma;
        const hn::CappedTag<float, 16> dt;
        const int N = static_cast<int>(hn::Lanes(dt));
        const auto vsig = hn::Set(dt, sig2);
        const auto vone = hn::Set(dt, 1.f);
        for (int i = 0; i < group; ++i) {
            float* row = cube + static_cast<size_t>(i) * area;
            const float* rr = refw + static_cast<size_t>(i) * area;
            auto accw = hn::Zero(dt);
            int f = 0;
            for (; f + N <= area; f += N) {
                auto r = hn::LoadU(dt, rr + f);
                auto r2 = hn::Mul(r, r);
                auto w = hn::Div(r2, hn::Add(r2, vsig));
                if (i == 0 && f == 0) {
                    w = hn::IfThenElse(hn::FirstN(dt, 1), vone, w);
                }
                hn::StoreU(hn::Mul(hn::LoadU(dt, row + f), w), dt, row + f);
                accw = hn::MulAdd(w, w, accw);
            }
            w2sum += hn::ReduceSum(dt, accw);
            for (; f < area; ++f) {
                float w = 1.f;
                if (!(f == 0 && i == 0)) {
                    const float r = rr[f];
                    const float r2 = r * r;
                    w = r2 / (r2 + sig2);
                }
                row[f] *= w;
                w2sum += w * w;
            }
        }
    } else {
        const float thr = kBmHardLambda * sigma;
        const hn::CappedTag<float, 16> dt;
        const int N = static_cast<int>(hn::Lanes(dt));
        const auto vthr = hn::Set(dt, thr);
        for (int i = 0; i < group; ++i) {
            float* row = cube + static_cast<size_t>(i) * area;
            int f = 0;
            for (; f + N <= area; f += N) {
                auto v = hn::LoadU(dt, row + f);
                auto kill = hn::Lt(hn::Abs(v), vthr);
                if (i == 0 && f == 0) {
                    kill = hn::And(kill, hn::Not(hn::FirstN(dt, 1)));
                }
                hn::StoreU(hn::IfThenZeroElse(kill, v), dt, row + f);
                kept += N - static_cast<int>(hn::CountTrue(dt, kill));
            }
            for (; f < area; ++f) {
                float& c = row[f];
                if (!(f == 0 && i == 0) && std::fabs(c) < thr) {
                    c = 0.f;
                } else {
                    ++kept;
                }
            }
        }
    }
    DctLines(cube, group, 1, area, area, true);
    dct2(cube, true);
    if (!inplace) {
        for (int i = 0; i < kk; ++i) {
            std::memcpy(patches + i * lda, cube + static_cast<size_t>(i) * area,
                        static_cast<size_t>(area) * sizeof(float));
        }
    }
    if (weight_out) {
        if (wiener) {
            *weight_out = 1.f / std::max(w2sum, 1e-12f);
        } else {
            *weight_out = 1.f / static_cast<float>(std::max(kept, 1));
        }
    }
}

#if HWY_MAX_BYTES >= 32
// FFTW 3.3.9 e10_8 / e01_8, same layout as bm3dcpu: G[g*8+r] is row r of patch g, lanes = x.
// Unnormalized: 3D roundtrip scales by 4096. Inverse is DCT-III.
template <bool kForward>
HWY_INLINE void Dct8Fftw(V8 block[8]) {
    const D8 d8;
    const auto kp414 = hn::Set(d8, 0.414213562373095048801688724209698078569671875f);
    const auto kp1847 = hn::Set(d8, 1.847759065022573512256366378793576573644833252f);
    const auto kp198 = hn::Set(d8, 0.198912367379658006911597622644676228597850501f);
    const auto kp1961 = hn::Set(d8, 1.961570560806460898252364472268478073947867462f);
    const auto kp1414 = hn::Set(d8, 1.414213562373095048801688724209698078569671875f);
    const auto kp668 = hn::Set(d8, 0.668178637919298919997757686523080761552472251f);
    const auto kp1662 = hn::Set(d8, 1.662939224605090474157576755235811513477121624f);
    const auto kp707 = hn::Set(d8, 0.707106781186547524400844362104849039284835938f);
    if constexpr (kForward) {
        const auto t1 = block[0];
        const auto t2 = block[7];
        const auto t3 = hn::Sub(t1, t2);
        const auto tj = hn::Add(t1, t2);
        const auto tc = block[4];
        const auto td = block[3];
        const auto te = hn::Sub(tc, td);
        const auto tk = hn::Add(tc, td);
        const auto t4 = block[2];
        const auto t5 = block[5];
        const auto t6 = hn::Sub(t4, t5);
        const auto t7 = block[1];
        const auto t8 = block[6];
        const auto t9 = hn::Sub(t7, t8);
        const auto ta = hn::Add(t6, t9);
        const auto tn = hn::Add(t7, t8);
        const auto tf = hn::Sub(t6, t9);
        const auto tm = hn::Add(t4, t5);
        const auto tb = hn::NegMulAdd(kp707, ta, t3);
        const auto tg = hn::NegMulAdd(kp707, tf, te);
        block[3] = hn::Mul(kp1662, hn::MulAdd(kp668, tg, tb));
        block[5] = hn::Neg(hn::Mul(kp1662, hn::NegMulAdd(kp668, tb, tg)));
        const auto tp = hn::Add(tj, tk);
        const auto tq = hn::Add(tm, tn);
        block[4] = hn::Mul(kp1414, hn::Sub(tp, tq));
        block[0] = hn::Mul(kp1414, hn::Add(tp, tq));
        const auto th = hn::MulAdd(kp707, ta, t3);
        const auto ti = hn::MulAdd(kp707, tf, te);
        block[1] = hn::Mul(kp1961, hn::NegMulAdd(kp198, ti, th));
        block[7] = hn::Mul(kp1961, hn::MulAdd(kp198, th, ti));
        const auto tl = hn::Sub(tj, tk);
        const auto to = hn::Sub(tm, tn);
        block[2] = hn::Mul(kp1847, hn::NegMulAdd(kp414, to, tl));
        block[6] = hn::Mul(kp1847, hn::MulAdd(kp414, tl, to));
    } else {
        const auto t1 = hn::Mul(kp1414, block[0]);
        const auto t2 = block[4];
        const auto t3 = hn::MulAdd(kp1414, t2, t1);
        const auto tj = hn::NegMulAdd(kp1414, t2, t1);
        const auto t4 = block[2];
        const auto t5 = block[6];
        const auto t6 = hn::MulAdd(kp414, t5, t4);
        const auto tk = hn::Sub(hn::Mul(kp414, t4), t5);
        const auto t8 = block[1];
        const auto td = block[7];
        const auto t9 = block[5];
        const auto ta = block[3];
        const auto tb = hn::Add(t9, ta);
        const auto te = hn::Sub(ta, t9);
        const auto tc = hn::MulAdd(kp707, tb, t8);
        const auto tn = hn::NegMulAdd(kp707, te, td);
        const auto tf = hn::MulAdd(kp707, te, td);
        const auto tm = hn::NegMulAdd(kp707, tb, t8);
        const auto t7 = hn::MulAdd(kp1847, t6, t3);
        const auto tg = hn::MulAdd(kp198, tf, tc);
        block[7] = hn::NegMulAdd(kp1961, tg, t7);
        block[0] = hn::MulAdd(kp1961, tg, t7);
        const auto tp = hn::NegMulAdd(kp1847, tk, tj);
        const auto tq = hn::MulAdd(kp668, tm, tn);
        block[5] = hn::NegMulAdd(kp1662, tq, tp);
        block[2] = hn::MulAdd(kp1662, tq, tp);
        const auto th = hn::NegMulAdd(kp1847, t6, t3);
        const auto ti = hn::NegMulAdd(kp198, tc, tf);
        block[3] = hn::NegMulAdd(kp1961, ti, th);
        block[4] = hn::MulAdd(kp1961, ti, th);
        const auto tl = hn::MulAdd(kp1847, tk, tj);
        const auto to = hn::NegMulAdd(kp668, tn, tm);
        block[6] = hn::NegMulAdd(kp1662, to, tl);
        block[1] = hn::MulAdd(kp1662, to, tl);
    }
}

template <bool kForward, int kStride, int kHowMany, int kHowManyStride>
HWY_INLINE void TransformPack8(V8* data) {
    for (int iter = 0; iter < kHowMany; ++iter, data += kHowManyStride) {
        if constexpr (kStride == 1) {
            Dct8Fftw<kForward>(data);
        } else {
            V8 v[8];
            for (int i = 0; i < 8; ++i) {
                v[i] = data[i * kStride];
            }
            Dct8Fftw<kForward>(v);
            for (int i = 0; i < 8; ++i) {
                data[i * kStride] = v[i];
            }
        }
    }
}

HWY_INLINE void TransposePack8(V8* data) {
    const D8 d8;
    for (int g = 0; g < 8; ++g) {
        V8 t[8];
        Transpose8x8Vec(d8, data + g * 8, t);
        for (int r = 0; r < 8; ++r) {
            data[g * 8 + r] = t[r];
        }
    }
}

// Per-patch 2D DCT: row transform, in-register 8x8 transpose, column transform.
// Avoids TransposePack8's store/reload of the whole 8-patch cube between the
// two spatial passes. Group-axis DCT stays a separate packed pass.
template <bool kForward>
HWY_INLINE void Fftw2dInReg(V8* patch) {
    const D8 d8;
    Dct8Fftw<kForward>(patch);
    V8 t[8];
    Transpose8x8Vec(d8, patch, t);
    Dct8Fftw<kForward>(t);
    for (int r = 0; r < 8; ++r) {
        patch[r] = t[r];
    }
}

HWY_INLINE void Fftw3dFwd(V8* data) {
    for (int g = 0; g < 8; ++g) {
        Fftw2dInReg<true>(data + g * 8);
    }
    TransformPack8<true, 8, 8, 1>(data);
}

HWY_INLINE void Fftw3dInv(V8* data) {
    for (int g = 0; g < 8; ++g) {
        Fftw2dInReg<false>(data + g * 8);
    }
    TransformPack8<false, 8, 8, 1>(data);
}

HWY_INLINE float HardThreshFftw(V8 G[64], float sigma) {
    const D8 d8;
    HWY_ALIGN float mask_mem[8] = {0.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    const auto thr_mask = hn::Load(d8, mask_mem);
    const auto vsig = hn::Set(d8, sigma);
    int kept = 0;
    for (int i = 0; i < 64; ++i) {
        const auto val = G[i];
        const auto thr = (i == 0) ? hn::Mul(vsig, thr_mask) : vsig;
        const auto kill = hn::Lt(hn::Abs(val), thr);
        G[i] = hn::IfThenZeroElse(kill, val);
        kept += 8 - static_cast<int>(hn::CountTrue(d8, kill));
    }
    return 1.f / static_cast<float>(std::max(kept, 1));
}

HWY_INLINE float WienerFftw(V8 G[64], const V8 R[64], float sigma) {
    const D8 d8;
    const auto vsig2 = hn::Set(d8, sigma * sigma);
    const auto vone = hn::Set(d8, 1.f);
    auto accw = hn::Zero(d8);
    for (int i = 0; i < 64; ++i) {
        const auto r = R[i];
        const auto r2 = hn::Mul(r, r);
        auto w = hn::Div(r2, hn::Add(r2, vsig2));
        if (i == 0) {
            w = hn::IfThenElse(hn::FirstN(d8, 1), vone, w);
        }
        accw = hn::MulAdd(w, w, accw);
        G[i] = hn::Mul(G[i], w);
    }
    return 1.f / std::max(hn::ReduceSum(d8, accw), 1e-12f);
}
#endif

void Bm3dFilter8(const float* src, int sstride, const Match* matches, int k, float sigma, bool wiener,
                 const float* ref, int rstride, float* num, float* den, int dstride, int width, int height) {
#if HWY_MAX_BYTES >= 32
    const D8 d8;
    const int kk = std::min(std::max(k, 1), 8);
    bool all_inside = true;
    for (int g = 0; g < kk; ++g) {
        const int x = matches[g].x;
        const int y = matches[g].y;
        all_inside = all_inside && x >= 0 && y >= 0 && x + 8 <= width && y + 8 <= height;
    }
    V8 G[64];
    if (all_inside) {
        for (int g = 0; g < kk; ++g) {
            LoadPatch8Inside(src, sstride, matches[g].x, matches[g].y, G + g * 8);
        }
    } else {
        for (int g = 0; g < kk; ++g) {
            LoadPatch8(src, sstride, matches[g].x, matches[g].y, width, height, G + g * 8);
        }
    }
    for (int g = kk; g < 8; ++g) {
        for (int r = 0; r < 8; ++r) {
            G[g * 8 + r] = hn::Zero(d8);
        }
    }
    // bm3dcpu: user sigma * (3/4)/255 * 64 * (hard ? 2.7 : 1). Host already /255.
    const float s = sigma * 0.75f * 64.f;
    float wgt = 1.f;
    if (wiener && ref) {
        V8 R[64];
        if (all_inside) {
            for (int g = 0; g < kk; ++g) {
                LoadPatch8Inside(ref, rstride, matches[g].x, matches[g].y, R + g * 8);
            }
        } else {
            for (int g = 0; g < kk; ++g) {
                LoadPatch8(ref, rstride, matches[g].x, matches[g].y, width, height, R + g * 8);
            }
        }
        for (int g = kk; g < 8; ++g) {
            for (int r = 0; r < 8; ++r) {
                R[g * 8 + r] = hn::Zero(d8);
            }
        }
        Fftw3dFwd(G);
        Fftw3dFwd(R);
        wgt = WienerFftw(G, R, s);
        Fftw3dInv(G);
    } else {
        Fftw3dFwd(G);
        wgt = HardThreshFftw(G, kBmHardLambda * s);
        Fftw3dInv(G);
    }
    if (all_inside) {
        for (int g = 0; g < kk; ++g) {
            AccPatch8Inside(num, den, dstride, matches[g].x, matches[g].y, G + g * 8, wgt / 4096.f, wgt);
        }
    } else {
        for (int g = 0; g < kk; ++g) {
            AccPatch8(num, den, dstride, matches[g].x, matches[g].y, width, height, G + g * 8, wgt / 4096.f, wgt);
        }
    }
#else
    (void)src;
    (void)sstride;
    (void)matches;
    (void)k;
    (void)sigma;
    (void)wiener;
    (void)ref;
    (void)rstride;
    (void)num;
    (void)den;
    (void)dstride;
    (void)width;
    (void)height;
#endif
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(Dct1d);
HWY_EXPORT(Idct1d);
HWY_EXPORT(Dct2d);
HWY_EXPORT(Idct2d);
HWY_EXPORT(Dct8_1d);
HWY_EXPORT(Idct8_1d);
HWY_EXPORT(Dct8_2d);
HWY_EXPORT(Idct8_2d);
HWY_EXPORT(TransformLines);
HWY_EXPORT(Bm3dFilterGroup);
HWY_EXPORT(Bm3dFilter8);

void dct_1d(const float* in, float* out, int n) {
    HWY_DYNAMIC_DISPATCH(Dct1d)(in, out, n);
}
void idct_1d(const float* in, float* out, int n) {
    HWY_DYNAMIC_DISPATCH(Idct1d)(in, out, n);
}
void dct_2d(float* block, int n) {
    HWY_DYNAMIC_DISPATCH(Dct2d)(block, n);
}
void idct_2d(float* block, int n) {
    HWY_DYNAMIC_DISPATCH(Idct2d)(block, n);
}
void dct8_1d(const float in[8], float out[8]) {
    HWY_DYNAMIC_DISPATCH(Dct8_1d)(in, out);
}
void idct8_1d(const float in[8], float out[8]) {
    HWY_DYNAMIC_DISPATCH(Idct8_1d)(in, out);
}
void dct8_2d(float block[64]) {
    HWY_DYNAMIC_DISPATCH(Dct8_2d)(block);
}
void idct8_2d(float block[64]) {
    HWY_DYNAMIC_DISPATCH(Idct8_2d)(block);
}

void dct_lines(float* base, int n, int line_stride, int sample_stride, int count, bool inverse) {
    HWY_DYNAMIC_DISPATCH(TransformLines)(base, n, line_stride, sample_stride, count, inverse);
}

void bm3d_filter_group(float* patches, int lda, int group, int k, int block, float sigma, bool wiener,
                       const float* ref_patches, float* weight_out, float* work) {
    HWY_DYNAMIC_DISPATCH(Bm3dFilterGroup)(patches, lda, group, k, block, sigma, wiener, ref_patches, weight_out, work);
}

void bm3d_filter8(const float* src, int sstride, const Match* matches, int k, float sigma, bool wiener,
                  const float* ref, int rstride, float* num, float* den, int dstride, int width, int height) {
    HWY_DYNAMIC_DISPATCH(Bm3dFilter8)(src, sstride, matches, k, sigma, wiener, ref, rstride, num, den, dstride, width,
                                     height);
}

}  // namespace nss
#endif
