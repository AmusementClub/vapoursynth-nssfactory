// Included inside nss::HWY_NAMESPACE after highway.h. No include guard: foreach_target
// re-includes the TU once per ISA.
namespace {
constexpr float kHaarInvSqrt2 = 0.7071067811865475244f;

static inline void Haar1dN(const float* in, float* out, int n) {
    if (n == 1) {
        out[0] = in[0];
        return;
    }
    float a[16];
    float b[16];
    for (int i = 0; i < n; ++i) {
        a[i] = in[i];
    }
    for (int len = n; len >= 2; len /= 2) {
        const int h = len / 2;
        for (int i = 0; i < h; ++i) {
            const float x0 = a[2 * i];
            const float x1 = a[2 * i + 1];
            b[i] = (x0 + x1) * kHaarInvSqrt2;
            b[h + i] = (x0 - x1) * kHaarInvSqrt2;
        }
        for (int i = 0; i < len; ++i) {
            a[i] = b[i];
        }
    }
    for (int i = 0; i < n; ++i) {
        out[i] = a[i];
    }
}

static inline void IHaar1dN(const float* in, float* out, int n) {
    if (n == 1) {
        out[0] = in[0];
        return;
    }
    float a[16];
    float b[16];
    for (int i = 0; i < n; ++i) {
        a[i] = in[i];
    }
    for (int len = 2; len <= n; len *= 2) {
        const int h = len / 2;
        for (int i = 0; i < h; ++i) {
            const float s = a[i];
            const float dlt = a[h + i];
            b[2 * i] = (s + dlt) * kHaarInvSqrt2;
            b[2 * i + 1] = (s - dlt) * kHaarInvSqrt2;
        }
        for (int i = 0; i < len; ++i) {
            a[i] = b[i];
        }
    }
    for (int i = 0; i < n; ++i) {
        out[i] = a[i];
    }
}

static inline void Haar4Vec(hn::Vec<hn::ScalableTag<float>>& v0, hn::Vec<hn::ScalableTag<float>>& v1,
                            hn::Vec<hn::ScalableTag<float>>& v2, hn::Vec<hn::ScalableTag<float>>& v3, bool inv) {
    const hn::ScalableTag<float> d;
    const auto s = hn::Set(d, kHaarInvSqrt2);
    if (!inv) {
        const auto b0 = hn::Mul(hn::Add(v0, v1), s);
        const auto b1 = hn::Mul(hn::Add(v2, v3), s);
        const auto b2 = hn::Mul(hn::Sub(v0, v1), s);
        const auto b3 = hn::Mul(hn::Sub(v2, v3), s);
        v0 = hn::Mul(hn::Add(b0, b1), s);
        v1 = hn::Mul(hn::Sub(b0, b1), s);
        v2 = b2;
        v3 = b3;
    } else {
        const auto t0 = hn::Mul(hn::Add(v0, v1), s);
        const auto t1 = hn::Mul(hn::Sub(v0, v1), s);
        const auto o0 = hn::Mul(hn::Add(t0, v2), s);
        const auto o1 = hn::Mul(hn::Sub(t0, v2), s);
        const auto o2 = hn::Mul(hn::Add(t1, v3), s);
        const auto o3 = hn::Mul(hn::Sub(t1, v3), s);
        v0 = o0;
        v1 = o1;
        v2 = o2;
        v3 = o3;
    }
}

static void HaarColsQ4(float* mat, int n, bool inverse) {
    const hn::ScalableTag<float> d;
    const int L = static_cast<int>(hn::Lanes(d));
    int c = 0;
    for (; c + L <= n; c += L) {
        float* p = mat + c * 4;
        hn::Vec<hn::ScalableTag<float>> v0, v1, v2, v3;
        hn::LoadInterleaved4(d, p, v0, v1, v2, v3);
        Haar4Vec(v0, v1, v2, v3, inverse);
        hn::StoreInterleaved4(v0, v1, v2, v3, d, p);
    }
    for (; c < n; ++c) {
        float* col = mat + c * 4;
        if (inverse) {
            IHaar1dN(col, col, 4);
        } else {
            Haar1dN(col, col, 4);
        }
    }
}

static void HaarColsQ2(float* mat, int n, bool inverse) {
    (void)inverse;
    const hn::ScalableTag<float> d;
    const int L = static_cast<int>(hn::Lanes(d));
    const auto s = hn::Set(d, kHaarInvSqrt2);
    int c = 0;
    for (; c + L <= n; c += L) {
        float* p = mat + c * 2;
        hn::Vec<hn::ScalableTag<float>> v0, v1;
        hn::LoadInterleaved2(d, p, v0, v1);
        const auto o0 = hn::Mul(hn::Add(v0, v1), s);
        const auto o1 = hn::Mul(hn::Sub(v0, v1), s);
        hn::StoreInterleaved2(o0, o1, d, p);
    }
    for (; c < n; ++c) {
        Haar1dN(mat + c * 2, mat + c * 2, 2);
    }
}

static void Haar2dFast(float* mat, int q, int n, bool inverse);

#if HWY_MAX_BYTES >= 64
static inline hn::Vec<hn::ScalableTag<float>> Haar16Zip8(
    hn::Vec<hn::Half<hn::ScalableTag<float>>> lo, hn::Vec<hn::Half<hn::ScalableTag<float>>> hi) {
    const hn::ScalableTag<float> d16;
    alignas(64) static constexpr int32_t kZip[16] = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};
    return hn::TableLookupLanes(hn::Combine(d16, hi, lo), hn::SetTableIndices(d16, kZip));
}

static inline hn::Vec<hn::Half<hn::ScalableTag<float>>> Haar8Zip4(
    hn::Vec<hn::Half<hn::Half<hn::ScalableTag<float>>>> lo,
    hn::Vec<hn::Half<hn::Half<hn::ScalableTag<float>>>> hi) {
    const hn::Half<hn::ScalableTag<float>> d8;
    alignas(32) static constexpr int32_t kZip[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    return hn::TableLookupLanes(hn::Combine(d8, hi, lo), hn::SetTableIndices(d8, kZip));
}

static inline hn::Vec<hn::ScalableTag<float>> Haar16VecFwd(hn::Vec<hn::ScalableTag<float>> v) {
    const hn::ScalableTag<float> d16;
    const hn::Half<hn::ScalableTag<float>> d8;
    const hn::Half<hn::Half<hn::ScalableTag<float>>> d4;
    const auto s16 = hn::Set(d16, kHaarInvSqrt2);
    const auto s8 = hn::Set(d8, kHaarInvSqrt2);
    auto even = hn::ConcatEven(d16, v, v);
    auto odd = hn::ConcatOdd(d16, v, v);
    v = hn::Combine(d16, hn::LowerHalf(hn::Mul(hn::Sub(even, odd), s16)),
                    hn::LowerHalf(hn::Mul(hn::Add(even, odd), s16)));
    auto lo8 = hn::LowerHalf(d8, v);
    const auto hi8 = hn::UpperHalf(d8, v);
    const auto even8 = hn::ConcatEven(d8, lo8, lo8);
    const auto odd8 = hn::ConcatOdd(d8, lo8, lo8);
    lo8 = hn::Combine(d8, hn::LowerHalf(hn::Mul(hn::Sub(even8, odd8), s8)),
                      hn::LowerHalf(hn::Mul(hn::Add(even8, odd8), s8)));
    float p[4];
    hn::StoreU(hn::LowerHalf(lo8), d4, p);
    Haar1dN(p, p, 4);
    lo8 = hn::Combine(d8, hn::UpperHalf(d4, lo8), hn::LoadU(d4, p));
    return hn::Combine(d16, hi8, lo8);
}

static inline hn::Vec<hn::ScalableTag<float>> Haar16VecInv(hn::Vec<hn::ScalableTag<float>> v) {
    const hn::ScalableTag<float> d16;
    const hn::Half<hn::ScalableTag<float>> d8;
    const hn::Half<hn::Half<hn::ScalableTag<float>>> d4;
    const auto s16 = hn::Set(d16, kHaarInvSqrt2);
    const auto s4 = hn::Set(d4, kHaarInvSqrt2);
    auto lo8 = hn::LowerHalf(d8, v);
    const auto hi8 = hn::UpperHalf(d8, v);
    float p[4];
    hn::StoreU(hn::LowerHalf(lo8), d4, p);
    IHaar1dN(p, p, 4);
    const auto lo4 = hn::LoadU(d4, p);
    const auto hi4 = hn::UpperHalf(d4, lo8);
    lo8 = Haar8Zip4(hn::Mul(hn::Add(lo4, hi4), s4), hn::Mul(hn::Sub(lo4, hi4), s4));
    const auto s8 = hn::Set(d8, kHaarInvSqrt2);
    return Haar16Zip8(hn::Mul(hn::Add(lo8, hi8), s8), hn::Mul(hn::Sub(lo8, hi8), s8));
}
#endif

static void Haar16InVec(hn::Vec<hn::ScalableTag<float>>& v, bool inverse) {
#if HWY_MAX_BYTES >= 64
    if (static_cast<int>(hn::Lanes(hn::ScalableTag<float>())) == 16) {
        v = inverse ? Haar16VecInv(v) : Haar16VecFwd(v);
        return;
    }
#endif
    const hn::ScalableTag<float> d;
    float tmp[16];
    hn::StoreU(v, d, tmp);
    if (inverse) {
        IHaar1dN(tmp, tmp, 16);
    } else {
        Haar1dN(tmp, tmp, 16);
    }
    v = hn::LoadU(d, tmp);
}

static void Haar2d_4x16(float* mat, bool inverse) {
    const hn::ScalableTag<float> d;
    if (static_cast<int>(hn::Lanes(d)) == 16) {
        hn::Vec<hn::ScalableTag<float>> v0, v1, v2, v3;
        hn::LoadInterleaved4(d, mat, v0, v1, v2, v3);
        if (!inverse) {
            Haar4Vec(v0, v1, v2, v3, false);
            Haar16InVec(v0, false);
            Haar16InVec(v1, false);
            Haar16InVec(v2, false);
            Haar16InVec(v3, false);
        } else {
            Haar16InVec(v0, true);
            Haar16InVec(v1, true);
            Haar16InVec(v2, true);
            Haar16InVec(v3, true);
            Haar4Vec(v0, v1, v2, v3, true);
        }
        hn::StoreInterleaved4(v0, v1, v2, v3, d, mat);
        return;
    }
    Haar2dFast(mat, 4, 16, inverse);
}

static void Haar2dFast(float* mat, int q, int n, bool inverse) {
    auto cols = [&]() {
        if (q == 4) {
            HaarColsQ4(mat, n, inverse);
            return;
        }
        if (q == 2) {
            HaarColsQ2(mat, n, inverse);
            return;
        }
        for (int c = 0; c < n; ++c) {
            float* col = mat + c * q;
            if (inverse) {
                IHaar1dN(col, col, q);
            } else {
                Haar1dN(col, col, q);
            }
        }
    };
    auto rows = [&]() {
        float tmp[16];
        for (int r = 0; r < q; ++r) {
            for (int c = 0; c < n; ++c) {
                tmp[c] = mat[r + c * q];
            }
            if (inverse) {
                IHaar1dN(tmp, tmp, n);
            } else {
                Haar1dN(tmp, tmp, n);
            }
            for (int c = 0; c < n; ++c) {
                mat[r + c * q] = tmp[c];
            }
        }
    };
    if (inverse) {
        rows();
        cols();
    } else {
        cols();
        rows();
    }
}

}  // namespace
