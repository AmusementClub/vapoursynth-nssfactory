// Included in a Highway target namespace; intentionally no include guard.
#if HWY_MAX_BYTES >= 64
template <int B>
HWY_NOINLINE void SsdRowSplit8(const float* self, const float* candidates, int stride, int count,
                             float* distances) {
    const hn::FixedTag<float, 8> d;
    const hn::FixedTag<float, 16> d16;
    hn::Vec<decltype(d)> lo[B], hi[B];
    for (int r = 0; r < B; ++r) {
        lo[r] = hn::LoadU(d, self + r * stride);
        hi[r] = hn::LoadN(d, self + r * stride + 8, B - 8);
    }
    for (int x = 0; x < count; x += 2) {
        auto a = hn::Zero(d), b = hn::Zero(d);
        auto c = hn::Zero(d), e = hn::Zero(d);
        const bool pair = x + 1 < count;
        for (int r = 0; r < B; ++r) {
            const float* p = candidates + x + r * stride;
            const auto dl = hn::Sub(lo[r], hn::LoadU(d, p));
            const auto dh = hn::Sub(hi[r], hn::LoadN(d, p + 8, B - 8));
            a = hn::MulAdd(dl, dl, a);
            b = hn::MulAdd(dh, dh, b);
            if (pair) {
                const auto el = hn::Sub(lo[r], hn::LoadU(d, p + 1));
                const auto eh = hn::Sub(hi[r], hn::LoadN(d, p + 9, B - 8));
                c = hn::MulAdd(el, el, c);
                e = hn::MulAdd(eh, eh, e);
            }
        }
        // Keep Highway's 16-lane reduction tree, including its compiler-specific
        // quarter ordering; reducing the sum of two halves can change ties.
        distances[x] = hn::ReduceSum(d16, hn::Combine(d16, b, a));
        if (pair) distances[x + 1] = hn::ReduceSum(d16, hn::Combine(d16, e, c));
    }
}

template <int B>
HWY_NOINLINE void SsdRowStream(const float* self, const float* candidates, int stride, int count,
                             float* distances) {
    const hn::FixedTag<float, 16> d;
    for (int x = 0; x < count; x += 2) {
        auto a = hn::Zero(d), b = hn::Zero(d);
        const bool pair = x + 1 < count;
        for (int r = 0; r < B; ++r) {
            const auto ref = hn::LoadN(d, self + r * stride, B);
            const float* p = candidates + x + r * stride;
            const auto da = hn::Sub(ref, hn::LoadN(d, p, B));
            a = hn::MulAdd(da, da, a);
            if (pair) {
                const auto db = hn::Sub(ref, hn::LoadN(d, p + 1, B));
                b = hn::MulAdd(db, db, b);
            }
        }
        distances[x] = hn::ReduceSum(d, a);
        if (pair) distances[x + 1] = hn::ReduceSum(d, b);
    }
}
#endif
