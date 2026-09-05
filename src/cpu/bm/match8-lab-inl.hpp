// Frozen SpatialMatch8 experiment harness. Kept out of ordinary builds.
// Capture is untimed; replay preserves candidate order and signals fallback.
template <int Variant, bool Replay = false, bool Capture = false>
static int SpatialMatch8Lab(const float* ref, int stride, int width, int height, int cx, int cy, int bm_range,
                         Match* out, const detail::MatchReplayItem* replay = nullptr,
                         int replay_count = 0, detail::MatchReplayItem* trace = nullptr) {
    const hn::FixedTag<float, 8> df;
    const hn::FixedTag<std::int32_t, 8> di;
    const hn::RebindToUnsigned<decltype(di)> du;
    const int max_x = width - 8;
    const int max_y = height - 8;
    const int top = std::max(cy - bm_range, 0);
    const int bottom = std::min(cy + bm_range, max_y);
    const int left = std::max(cx - bm_range, 0);
    const int right = std::min(cx + bm_range, max_x);
    const float* self = ref ? ref + cy * stride + cx : nullptr;
    hn::Vec<decltype(df)> refb[8];
    if constexpr (!Replay) for (int i = 0; i < 8; ++i) {
        refb[i] = hn::LoadU(df, self + i * stride);
    }
    // Keep the self match in lane zero.  The explicit fill count below is
    // needed because this TU uses fast-math, where isfinite(infinity) is not
    // a reliable way to identify an unused lane.
    HWY_ALIGN float initial_errors[8] = {0.f,
                                         std::numeric_limits<float>::max(),
                                         std::numeric_limits<float>::max(),
                                         std::numeric_limits<float>::max(),
                                         std::numeric_limits<float>::max(),
                                         std::numeric_limits<float>::max(),
                                         std::numeric_limits<float>::max(),
                                         std::numeric_limits<float>::max()};
    auto errors = hn::Load(df, initial_errors);
    auto vx = hn::Set(di, cx);
    auto vy = hn::Set(di, cy);
    const auto iota = hn::Iota(di, 0);
    const auto one = hn::Set(di, 1);
    int filled = 1;
    float worst = std::numeric_limits<float>::max();
    auto insert = [&](int x, int y, hn::Vec<decltype(df)> vdist, int count) {
        const int pos = 8 - count;
        const auto posi = hn::Set(di, pos);
        alignas(64) static constexpr std::int32_t indices[8][8] = {
            {0,0,1,2,3,4,5,6}, {0,1,1,2,3,4,5,6}, {0,1,2,2,3,4,5,6}, {0,1,2,3,3,4,5,6},
            {0,1,2,3,4,4,5,6}, {0,1,2,3,4,5,5,6}, {0,1,2,3,4,5,6,6}, {0,1,2,3,4,5,6,7}};
        const auto src = Variant == 1 ? hn::Load(di, indices[pos])
            : hn::IfThenElse(hn::Gt(iota, posi), hn::Sub(iota, one), iota);
        const auto idxf = hn::IndicesFromVec(df, hn::BitCast(du, src));
        const auto idxi = hn::IndicesFromVec(di, hn::BitCast(du, src));
        const auto at = hn::RebindMask(df, hn::Eq(iota, posi));
        errors = hn::IfThenElse(at, vdist, hn::TableLookupLanes(errors, idxf));
        vx = hn::IfThenElse(hn::Eq(iota, posi), hn::Set(di, x), hn::TableLookupLanes(vx, idxi));
        vy = hn::IfThenElse(hn::Eq(iota, posi), hn::Set(di, y), hn::TableLookupLanes(vy, idxi));
    };
    int traced = 0;
    auto consider = [&](int x, int y, float dist) {
        if constexpr (Capture) trace[traced++] = {x, y, dist};
        if (x == cx && y == cy) {
            return;
        }
        if (filled < 8) {
            const auto vdist = hn::Set(df, dist);
            const int count = static_cast<int>(hn::CountTrue(df, hn::Lt(vdist, errors)));
            if (count != 0) {
                insert(x, y, vdist, count);
                ++filled;
            }
            if (filled == 8) {
                worst = hn::ExtractLane(errors, 7);
            }
        } else if (dist < worst) {
            const auto vdist = hn::Set(df, dist);
            const int count = static_cast<int>(hn::CountTrue(df, hn::Lt(vdist, errors)));
            insert(x, y, vdist, count);
            worst = hn::ExtractLane(errors, 7);
        }
    };
    auto fallback = [&]() {
        if constexpr (Replay) return -1;
        return detail::collect_spatial(ref, stride, width, height, cx, cy, 8, bm_range, 8, out,
                                       [](const float* a, const float* b, int st, int bs) {
                                           return SsdBlock(a, st, b, st, bs);
                                       });
    };
    if constexpr (Replay) {
        for (int i = 0; i < replay_count; ++i) consider(replay[i].x, replay[i].y, replay[i].distance);
    } else {
        for (int y = top; y <= bottom; ++y) {
            const float* row = ref + y * stride;
            int x = left;
            for (; x + 1 <= right; x += 2) {
                auto a00 = hn::Zero(df);
                auto a01 = hn::Zero(df);
                auto a02 = hn::Zero(df);
                auto a03 = hn::Zero(df);
                auto a10 = hn::Zero(df);
                auto a11 = hn::Zero(df);
                auto a12 = hn::Zero(df);
                auto a13 = hn::Zero(df);
                for (int i = 0; i < 8; i += 4) {
                    const auto d00 = hn::Sub(refb[i], hn::LoadU(df, row + x + i * stride));
                    const auto d01 = hn::Sub(refb[i + 1], hn::LoadU(df, row + x + (i + 1) * stride));
                    const auto d02 = hn::Sub(refb[i + 2], hn::LoadU(df, row + x + (i + 2) * stride));
                    const auto d03 = hn::Sub(refb[i + 3], hn::LoadU(df, row + x + (i + 3) * stride));
                    const auto d10 = hn::Sub(refb[i], hn::LoadU(df, row + x + 1 + i * stride));
                    const auto d11 = hn::Sub(refb[i + 1], hn::LoadU(df, row + x + 1 + (i + 1) * stride));
                    const auto d12 = hn::Sub(refb[i + 2], hn::LoadU(df, row + x + 1 + (i + 2) * stride));
                    const auto d13 = hn::Sub(refb[i + 3], hn::LoadU(df, row + x + 1 + (i + 3) * stride));
                    a00 = hn::MulAdd(d00, d00, a00);
                    a01 = hn::MulAdd(d01, d01, a01);
                    a02 = hn::MulAdd(d02, d02, a02);
                    a03 = hn::MulAdd(d03, d03, a03);
                    a10 = hn::MulAdd(d10, d10, a10);
                    a11 = hn::MulAdd(d11, d11, a11);
                    a12 = hn::MulAdd(d12, d12, a12);
                    a13 = hn::MulAdd(d13, d13, a13);
                }
                float dist0, dist1;
                const auto v0 = hn::Add(hn::Add(a00, a02), hn::Add(a01, a03));
                const auto v1 = hn::Add(hn::Add(a10, a12), hn::Add(a11, a13));
                if constexpr (Variant == 2) {
                    const hn::FixedTag<float, 4> d4;
                    const auto p0 = hn::Add(hn::LowerHalf(d4, v0), hn::UpperHalf(d4, v0));
                    const auto p1 = hn::Add(hn::LowerHalf(d4, v1), hn::UpperHalf(d4, v1));
                    const auto both = hn::Combine(df, p1, p0);
                    const auto pairs = hn::Add(both, hn::Reverse4(df, both));
                    const auto sums = hn::Add(pairs, hn::Reverse2(df, pairs));
                    dist0 = hn::GetLane(sums);
                    dist1 = hn::ExtractLane(sums, 4);
                } else {
                    dist0 = HSum8(v0);
                    dist1 = HSum8(v1);
                }
                consider(x, y, dist0);
                consider(x + 1, y, dist1);
            }
            if (x <= right) {
                auto acc0 = hn::Zero(df);
                auto acc1 = hn::Zero(df);
                auto acc2 = hn::Zero(df);
                auto acc3 = hn::Zero(df);
                for (int i = 0; i < 8; i += 4) {
                    const auto d0 = hn::Sub(refb[i], hn::LoadU(df, row + x + i * stride));
                    const auto d1 = hn::Sub(refb[i + 1], hn::LoadU(df, row + x + (i + 1) * stride));
                    const auto d2 = hn::Sub(refb[i + 2], hn::LoadU(df, row + x + (i + 2) * stride));
                    const auto d3 = hn::Sub(refb[i + 3], hn::LoadU(df, row + x + (i + 3) * stride));
                    acc0 = hn::MulAdd(d0, d0, acc0);
                    acc1 = hn::MulAdd(d1, d1, acc1);
                    acc2 = hn::MulAdd(d2, d2, acc2);
                    acc3 = hn::MulAdd(d3, d3, acc3);
                }
                const float dist = HSum8(hn::Add(hn::Add(acc0, acc2), hn::Add(acc1, acc3)));
                consider(x, y, dist);
            }
        }
    }
    std::uint32_t max_bits = hn::ReduceMax(du, hn::BitCast(du, errors));
#if defined(__GNUC__) || defined(__clang__)
    __asm__ volatile("" : "+r"(max_bits));
#else
    volatile std::uint32_t retained_max_bits = max_bits;
    max_bits = retained_max_bits;
#endif
    // Non-finite SSD results cannot pass the ordered lane comparison. The
    // fill count proves that enough candidates displaced the FLT_MAX
    // sentinels; the integer reduction keeps that conclusion valid under
    // -ffinite-math-only and also catches a retained NaN or infinity.
    const int candidate_count = (right - left + 1) * (bottom - top + 1) - 1;
    const int expected = 1 + std::min(7, std::max(candidate_count, 0));
    if (filled != expected || max_bits >= 0x7f800000u) {
        return fallback();
    }
    HWY_ALIGN float dists[8];
    HWY_ALIGN std::int32_t xs[8];
    HWY_ALIGN std::int32_t ys[8];
    hn::Store(errors, df, dists);
    hn::Store(vx, di, xs);
    hn::Store(vy, di, ys);
    int n = 0;
    for (int i = 0; i < filled; ++i) {
        out[n].x = xs[i];
        out[n].y = ys[i];
        out[n].t = 0;
        out[n].dist = dists[i];
        const int span = right - left + 1;
        out[n].ordinal = (xs[i] == cx && ys[i] == cy)
                             ? 0u
                             : static_cast<std::uint32_t>((ys[i] - top) * span + (xs[i] - left) + 1);
        ++n;
    }
    if (n == 0) {
        out[0].x = cx;
        out[0].y = cy;
        out[0].t = 0;
        out[0].dist = 0.f;
        return 1;
    }
    return n;
}

template <int Variant>
int Match8LabEntry(const float* ref, int stride, int w, int h, int x, int y, int range, Match* out) {
    if constexpr (Variant == 0) return SpatialMatch8(ref, stride, w, h, x, y, range, out);
    return SpatialMatch8Lab<Variant>(ref, stride, w, h, x, y, range, out);
}
template <int Variant>
int Match8CaptureEntry(const float* ref, int stride, int w, int h, int x, int y, int range, Match* out, detail::MatchReplayItem* trace) {
    return SpatialMatch8Lab<Variant, false, true>(ref, stride, w, h, x, y, range, out, nullptr, 0, trace);
}
template <int Variant>
int Match8ReplayEntry(const detail::MatchReplayItem* trace, int n, int w, int h, int x, int y, int range, Match* out) {
    return SpatialMatch8Lab<Variant, true>(nullptr, w, w, h, x, y, range, out, trace, n);
}
