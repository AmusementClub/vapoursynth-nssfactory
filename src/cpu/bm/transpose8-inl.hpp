// Inline shuffle network used by the DCT output sink so vector results can
// feed stores without a separate matrix reload. Re-included for each ISA.
static HWY_INLINE void Transpose8x8Inline(D8 d8, const V8 r[8], V8 c[8]) {
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
