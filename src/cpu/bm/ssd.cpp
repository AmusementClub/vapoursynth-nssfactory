// SPDX-License-Identifier: GPL-2.0-only
#include "nss/avx2_policy.hpp"
#include "nss/cpu_api.hpp"
#include "cpu/hwy_config.hpp"
#include "cpu/bm/matcher.hpp"
#ifdef NSS_BM_KERNEL_LAB
#include "cpu/bm/kernel_lab.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/bm/ssd.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

#include "cpu/bm/ssd_leaf-inl.hpp"

float SsdBlock(const float* a, int sa, const float* b, int sb, int block) {
    if (!a || !b || block < 1 || sa < block || sb < block) {
        // Invalid views are rejected by the public matcher; use a finite
        // sentinel here so this fast-math TU does not rely on an infinity
        // constant whose classification can be folded away.
        return std::numeric_limits<float>::max();
    }
    if (block == 8) {
        return Ssd8(a, sa, b, sb);
    }
#if HWY_MAX_BYTES >= 16
    if (block == 4) {
        return Ssd4(a, sa, b, sb);
    }
#endif
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    auto acc = hn::Zero(d);
    for (int y = 0; y < block; ++y) {
        const float* pa = a + y * sa;
        const float* pb = b + y * sb;
        int x = 0;
        for (; x + N <= block; x += N) {
            const auto diff = hn::Sub(hn::LoadU(d, pa + x), hn::LoadU(d, pb + x));
            acc = hn::MulAdd(diff, diff, acc);
        }
        const int rem = block - x;
        if (rem > 0) {
            const auto diff =
                hn::Sub(hn::LoadN(d, pa + x, static_cast<size_t>(rem)), hn::LoadN(d, pb + x, static_cast<size_t>(rem)));
            acc = hn::MulAdd(diff, diff, acc);
        }
    }
    return hn::ReduceSum(d, acc);
}

float SsdNch(const float* const* a, const int* sa, const float* const* b, const int* sb, int nch, int block) {
    if (!a || !sa || !b || !sb || nch < 1 || block < 1) {
        return std::numeric_limits<float>::max();
    }
#if HWY_MAX_BYTES >= 32
    if (nch == 3 && block == 8 && a[0] && a[1] && a[2] && b[0] && b[1] && b[2] && sa[0] >= 8 && sa[1] >= 8 &&
        sa[2] >= 8 && sb[0] >= 8 && sb[1] >= 8 && sb[2] >= 8) {
        const hn::FixedTag<float, 8> d;
        auto acc0 = hn::Zero(d);
        auto acc1 = hn::Zero(d);
        auto acc2 = hn::Zero(d);
        for (int y = 0; y < 8; ++y) {
            const auto d0 = hn::Sub(hn::LoadU(d, a[0] + y * sa[0]), hn::LoadU(d, b[0] + y * sb[0]));
            const auto d1 = hn::Sub(hn::LoadU(d, a[1] + y * sa[1]), hn::LoadU(d, b[1] + y * sb[1]));
            const auto d2 = hn::Sub(hn::LoadU(d, a[2] + y * sa[2]), hn::LoadU(d, b[2] + y * sb[2]));
            acc0 = hn::MulAdd(d0, d0, acc0);
            acc1 = hn::MulAdd(d1, d1, acc1);
            acc2 = hn::MulAdd(d2, d2, acc2);
        }
        return HSum8(hn::Add(acc0, hn::Add(acc1, acc2)));
    }
#endif
    float acc = 0.f;
    for (int c = 0; c < nch; ++c) {
        if (!a[c] || !b[c]) {
            continue;
        }
        acc += SsdBlock(a[c], sa[c], b[c], sb[c], block);
    }
    return acc;
}

static int clampi(int x, int lo, int hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

#if HWY_MAX_BYTES >= 32
// Sorted-ascending top-8 insert, same idea as bm3dcpu; only for block=8, group=8.
static int SpatialMatch8(const float* ref, int stride, int width, int height, int cx, int cy, int bm_range,
                         Match* out) {
    const hn::FixedTag<float, 8> df;
    const hn::FixedTag<std::int32_t, 8> di;
    const hn::RebindToUnsigned<decltype(di)> du;
    const int max_x = width - 8;
    const int max_y = height - 8;
    const int top = std::max(cy - bm_range, 0);
    const int bottom = std::min(cy + bm_range, max_y);
    const int left = std::max(cx - bm_range, 0);
    const int right = std::min(cx + bm_range, max_x);
    const float* self = ref + cy * stride + cx;
    hn::Vec<decltype(df)> refb[8];
    for (int i = 0; i < 8; ++i) {
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
        const auto src = hn::IfThenElse(hn::Gt(iota, posi), hn::Sub(iota, one), iota);
        const auto idxf = hn::IndicesFromVec(df, hn::BitCast(du, src));
        const auto idxi = hn::IndicesFromVec(di, hn::BitCast(du, src));
        const auto at = hn::RebindMask(df, hn::Eq(iota, posi));
        errors = hn::IfThenElse(at, vdist, hn::TableLookupLanes(errors, idxf));
        vx = hn::IfThenElse(hn::Eq(iota, posi), hn::Set(di, x), hn::TableLookupLanes(vx, idxi));
        vy = hn::IfThenElse(hn::Eq(iota, posi), hn::Set(di, y), hn::TableLookupLanes(vy, idxi));
    };
    auto consider = [&](int x, int y, float dist) {
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
        return detail::collect_spatial(ref, stride, width, height, cx, cy, 8, bm_range, 8, out,
                                       [](const float* a, const float* b, int st, int bs) {
                                           return SsdBlock(a, st, b, st, bs);
                                       });
    };
    for (int y = top; y <= bottom; ++y) {
        const float* row = ref + y * stride;
        int x = left;
        for (; x + 1 <= right; x += 2) {
#if NSS_BM_EXPERIMENT & 1024
            // The normalized DC coefficient is sum(a-b)/8, so its square
            // lower-bounds the complete 8x8 SSD. Double accumulation and an
            // outward margin preserve cutoff/tie competitors. This opt-in
            // prototype measures the bound's construction cost as well.
            if (filled == 8) {
                double sum0=0, sum1=0;
                for (int iy=0; iy<8; ++iy) for (int ix=0; ix<8; ++ix) {
                    const double a=static_cast<double>(self[iy*stride+ix]);
                    sum0 += a-static_cast<double>(row[x+iy*stride+ix]);
                    sum1 += a-static_cast<double>(row[x+1+iy*stride+ix]);
                }
                const double cutoff=static_cast<double>(worst)*(1.0+2e-5)+1e-12;
                if (sum0*sum0/64>cutoff && sum1*sum1/64>cutoff) continue;
            }
#endif
            auto a00 = hn::Zero(df);
            auto a01 = hn::Zero(df);
            auto a02 = hn::Zero(df);
            auto a03 = hn::Zero(df);
            auto a10 = hn::Zero(df);
            auto a11 = hn::Zero(df);
            auto a12 = hn::Zero(df);
            auto a13 = hn::Zero(df);
#if NSS_BM_EXPERIMENT & 32
            bool rejected=false;
#endif
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
#if NSS_BM_EXPERIMENT & 32
                if(i==0&&filled==8){
                    float p0=HSum8(hn::Add(hn::Add(a00,a02),hn::Add(a01,a03)));
                    float p1=HSum8(hn::Add(hn::Add(a10,a12),hn::Add(a11,a13)));
                    float bound=worst*(1.f+2e-5f)+1e-12f;
                    if(detail::finite_distance(p0)&&detail::finite_distance(p1)&&p0>bound&&p1>bound){rejected=true;break;}
                }
#endif
            }
#if NSS_BM_EXPERIMENT & 32
            if(rejected)continue;
#endif
            const float dist0 = HSum8(hn::Add(hn::Add(a00, a02), hn::Add(a01, a03)));
            const float dist1 = HSum8(hn::Add(hn::Add(a10, a12), hn::Add(a11, a13)));
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

#ifdef NSS_BM_KERNEL_LAB
#include "cpu/bm/match8-lab-inl.hpp"
#endif

// Same hoisted 8x8 SSD as SpatialMatch8, but top-k length follows group.
// group==8 keeps the lane-sorted path above; this is only the K-generic sibling.
#if (NSS_BM_EXPERIMENT & 1) || ((NSS_AVX2_EXPERIMENT & 1) && HWY_TARGET == HWY_AVX2)
template <class TopK>
#endif
static int SpatialMatch8Ssd(const float* ref, int stride, int width, int height, int cx, int cy, int bm_range,
                            int group, Match* out) {
    const hn::FixedTag<float, 8> df;
    const int wanted = std::min(group, kBmMaxGroup);
    const int top = std::max(cy - bm_range, 0);
    const int bottom = std::min(cy + bm_range, height - 8);
    const int left = std::max(cx - bm_range, 0);
    const int right = std::min(cx + bm_range, width - 8);
    const float* self = ref + cy * stride + cx;
    out[0] = Match{cx, cy, 0, 0.f, 0};
    if (wanted <= 1) {
        return 1;
    }

    hn::Vec<decltype(df)> refb[8];
    for (int i = 0; i < 8; ++i) {
        refb[i] = hn::LoadU(df, self + i * stride);
    }

    auto fallback = [&]() {
        return detail::collect_spatial(ref, stride, width, height, cx, cy, 8, bm_range, group, out,
                                       [](const float* a, const float* b, int st, int bs) {
                                           return SsdBlock(a, st, b, st, bs);
                                       });
    };

#if (NSS_BM_EXPERIMENT & 1) || ((NSS_AVX2_EXPERIMENT & 1) && HWY_TARGET == HWY_AVX2)
    TopK topk(out + 1, wanted - 1);
#else
    detail::CandidateTopK topk(out + 1, wanted - 1);
#endif
    bool nonfinite = false;
    const int span = right - left + 1;
    auto consider = [&](int x, int y, float dist) {
        if (nonfinite || (x == cx && y == cy)) {
            return;
        }
        if (!detail::finite_distance(dist)) {
            nonfinite = true;
            return;
        }
        const std::uint32_t ordinal = static_cast<std::uint32_t>((y - top) * span + (x - left) + 1);
        topk.add(Match{x, y, 0, dist, ordinal});
    };
    for (int y = top; y <= bottom && !nonfinite; ++y) {
        const float* row = ref + y * stride;
        int x = left;
        for (; x + 1 <= right && !nonfinite; x += 2) {
            // Same reduction tree as Ssd8 so group!=8 stays bit-exact vs collect_spatial.
            auto acc0 = hn::Zero(df);
            auto acc1 = hn::Zero(df);
            for (int i = 0; i < 8; ++i) {
                const auto d0 = hn::Sub(refb[i], hn::LoadU(df, row + x + i * stride));
                const auto d1 = hn::Sub(refb[i], hn::LoadU(df, row + x + 1 + i * stride));
                acc0 = hn::MulAdd(d0, d0, acc0);
                acc1 = hn::MulAdd(d1, d1, acc1);
            }
            consider(x, y, HSum8(acc0));
            consider(x + 1, y, HSum8(acc1));
        }
        if (x <= right && !nonfinite) {
            auto acc = hn::Zero(df);
            for (int i = 0; i < 8; ++i) {
                const auto d = hn::Sub(refb[i], hn::LoadU(df, row + x + i * stride));
                acc = hn::MulAdd(d, d, acc);
            }
            consider(x, y, HSum8(acc));
        }
    }
    if (nonfinite) {
        return fallback();
    }
    return 1 + topk.finish();
}

#endif

int SpatialMatch(const float* ref, int stride, int width, int height, int bx, int by, int block, int bm_range,
                 int group, Match* out, unsigned avx2_features) {
    const int max_x = width - block;
    const int max_y = height - block;
    if (!ref || !out || width < 1 || height < 1 || block < 1 || stride < width || max_x < 0 || max_y < 0 ||
        group < 1) {
        return 0;
    }
    const int cx = clampi(bx, 0, max_x);
    const int cy = clampi(by, 0, max_y);
#if HWY_MAX_BYTES >= 32
    if (block == 8 && group == 8) {
#if defined(NSS_BM_KERNEL_LAB) && defined(NSS_LAB_MATCH8_VARIANT)
        return Match8LabEntry<NSS_LAB_MATCH8_VARIANT>(ref, stride, width, height, cx, cy, std::max(bm_range, 0), out);
#else
        return SpatialMatch8(ref, stride, width, height, cx, cy, std::max(bm_range, 0), out);
#endif
    }
    if (block == 8) {
#if ((NSS_BM_EXPERIMENT & 1) && HWY_MAX_BYTES >= 64) || ((NSS_AVX2_EXPERIMENT & 1) && HWY_TARGET == HWY_AVX2)
        if (group >= 16
#if HWY_TARGET == HWY_AVX2
            && (avx2_features & 1)
#endif
        ) {
            return SpatialMatch8Ssd<detail::SpatialSortedTopK>(
                ref, stride, width, height, cx, cy, std::max(bm_range, 0), group, out);
        }
#endif
#if (NSS_BM_EXPERIMENT & 1) || ((NSS_AVX2_EXPERIMENT & 1) && HWY_TARGET == HWY_AVX2)
        return SpatialMatch8Ssd<detail::CandidateTopK>(
            ref, stride, width, height, cx, cy, std::max(bm_range, 0), group, out);
#else
        return SpatialMatch8Ssd(ref, stride, width, height, cx, cy, std::max(bm_range, 0), group, out);
#endif
    }
#endif
#if HWY_MAX_BYTES >= 16
    if (block == 4) {
        return detail::spatial_match4_fast(ref, stride, width, height, cx, cy, std::max(bm_range, 0), group, out, (avx2_features & 2) != 0);
    }
#endif
#ifndef NSS_DISABLE_BM3D_B12_FAST
    if (block == 12) {
        return detail::spatial_match12_fast(ref, stride, width, height, cx, cy, std::max(bm_range, 0), group, out, (avx2_features & 4) != 0);
    }
#endif
    if (block == 16) {
        return detail::spatial_match16_fast(ref, stride, width, height, cx, cy, std::max(bm_range, 0), group, out, (avx2_features & 8) != 0);
    }
    return detail::collect_spatial(ref, stride, width, height, cx, cy, block, bm_range, group, out,
                                   [](const float* a, const float* b, int st, int bs) {
                                       return SsdBlock(a, st, b, st, bs);
                                   });
}

// Block 8 for a run of n references at step 4 on one row ((x0 + 4 i, y), none
// clamped), sharing the work of overlapping blocks as the GPU square matcher
// does. Per displacement: column sums of the 8 rows along the run, tile sums
// of 4 columns, and reference i totals tiles i and i + 1. The distance sums the
// same squares in another order, so near ties may resolve differently from
// SpatialMatch (dyadic distances are exact either way). A vector test against
// each reference's current worst keeps most candidates out of the top-k.
// Returns 0, or -2 when a distance is nonfinite (the caller then takes the
// per-reference matcher for the whole run).
int SpatialSquareRun8(const float* ref, int stride, int width, int height, int x0, int y, int n, int bm_range,
                      int group, Match* out, int os, int* counts) {
    constexpr int kQ = 4, kB = 8;
    constexpr int kMaxRun = detail::kSquareMaxRun;
    if (!ref || !out || !counts || n < 1 || n > kMaxRun || x0 < 0 || y < 0 ||
        x0 + kQ * (n - 1) + kB > width || y + kB > height || stride < width) {
        return -2;
    }
    const hn::ScalableTag<float> d;
    const hn::RebindToUnsigned<decltype(d)> du;
    const int N = static_cast<int>(hn::Lanes(d));
    const int wanted = std::min(group, kBmMaxGroup);
    const int r = std::max(bm_range, 0);
    // Padded past the run by more than a vector's lanes, so whole-vector loads
    // and stores stay inside; those lanes are masked out below.
    constexpr int kPad = HWY_MAX_BYTES / sizeof(float) + 4;
    static_assert(HWY_MAX_BYTES / sizeof(float) <= 64, "mask bits must fit in 64 bits");
    HWY_ALIGN float col[kQ * (kMaxRun + 1) + kQ * kPad] = {};
    HWY_ALIGN float tile[kMaxRun + 1 + kPad] = {};
    HWY_ALIGN float dist[kMaxRun + kPad] = {};
    HWY_ALIGN float worst[kMaxRun + kPad];
    alignas(detail::SpatialSortedTopK) unsigned char top_bytes[kMaxRun * sizeof(detail::SpatialSortedTopK)];
    auto* tops = reinterpret_cast<detail::SpatialSortedTopK*>(top_bytes);
    int left[kMaxRun], span[kMaxRun], filled[kMaxRun];
    const int top = std::max(y - r, 0), bottom = std::min(y + r, height - kB);
    for (float& w : worst) w = std::numeric_limits<float>::max();
    for (int i = 0; i < n; ++i) {
        const int cx = x0 + kQ * i;
        left[i] = std::max(cx - r, 0);
        span[i] = std::min(cx + r, width - kB) - left[i] + 1;
        filled[i] = 0;
        out[static_cast<std::ptrdiff_t>(i) * os] = Match{cx, y, 0, 0.f, 0};
        new (&tops[i]) detail::SpatialSortedTopK(out + static_cast<std::ptrdiff_t>(i) * os + 1, wanted - 1);
    }
    if (wanted <= 1) {
        for (int i = 0; i < n; ++i) counts[i] = 1;
        return 0;
    }
    const int keep = wanted - 1;
    auto bad_bits = hn::Zero(du);
    for (int yy = top; yy <= bottom; ++yy) {
        const int dy = yy - y;
        for (int dx = -r; dx <= r; ++dx) {
            if (dx == 0 && dy == 0) continue;
            // references whose candidate (x0 + 4 i + dx, yy) lies in the image
            const int lim = width - kB - x0 - dx;
            if (lim < 0) continue;
            const int ilo = x0 + dx < 0 ? (kQ - 1 - (x0 + dx)) / kQ : 0;
            const int ihi = std::min(n - 1, lim / kQ);
            if (ilo > ihi) continue;
            const int c0 = kQ * ilo, c1 = kQ * (ihi + 2);
            for (int c = c0; c < c1; c += N) {
                const size_t cnt = static_cast<size_t>(std::min(N, c1 - c));
                auto acc0 = hn::Zero(d), acc1 = hn::Zero(d);
                for (int rr = 0; rr < kB; rr += 2) {
                    const float* pa = ref + static_cast<std::ptrdiff_t>(y + rr) * stride + x0 + c;
                    const float* pb = ref + static_cast<std::ptrdiff_t>(y + rr + dy) * stride + x0 + c + dx;
                    if (cnt == static_cast<size_t>(N)) {
                        const auto d0 = hn::Sub(hn::LoadU(d, pa), hn::LoadU(d, pb));
                        const auto d1 = hn::Sub(hn::LoadU(d, pa + stride), hn::LoadU(d, pb + stride));
                        acc0 = hn::MulAdd(d0, d0, acc0);
                        acc1 = hn::MulAdd(d1, d1, acc1);
                    } else {
                        const auto d0 = hn::Sub(hn::LoadN(d, pa, cnt), hn::LoadN(d, pb, cnt));
                        const auto d1 = hn::Sub(hn::LoadN(d, pa + stride, cnt), hn::LoadN(d, pb + stride, cnt));
                        acc0 = hn::MulAdd(d0, d0, acc0);
                        acc1 = hn::MulAdd(d1, d1, acc1);
                    }
                }
                hn::StoreU(hn::Add(acc0, acc1), d, col + c);
            }
            for (int j = ilo; j <= ihi + 1; j += N) {
                hn::Vec<decltype(d)> v0, v1, v2, v3;
                hn::LoadInterleaved4(d, col + kQ * j, v0, v1, v2, v3);
                hn::StoreU(hn::Add(hn::Add(v0, v1), hn::Add(v2, v3)), d, tile + j);
            }
            for (int i = ilo; i <= ihi; i += N) {
                const auto dv = hn::Add(hn::LoadU(d, tile + i), hn::LoadU(d, tile + i + 1));
                hn::StoreU(dv, d, dist + i);
                const auto valid = hn::FirstN(d, static_cast<size_t>(ihi - i + 1));
                // An integer test: this TU is fast-math, so a float isfinite could fold away.
                bad_bits = hn::Max(bad_bits,
                                   hn::IfThenElseZero(hn::RebindMask(du, valid), hn::BitCast(du, hn::Abs(dv))));
                const auto pass = hn::And(valid, hn::Le(dv, hn::LoadU(d, worst + i)));
                if (hn::AllFalse(d, pass)) continue;
                uint8_t mask_bytes[8] = {};
                hn::StoreMaskBits(d, pass, mask_bytes);
                uint64_t bits = 0;
                std::memcpy(&bits, mask_bytes, sizeof(bits));
                while (bits) {
                    const int q = i + static_cast<int>(hwy::Num0BitsBelowLS1Bit_Nonzero64(bits));
                    bits &= bits - 1;
                    const int cx = x0 + kQ * q + dx;
                    const auto ordinal = static_cast<std::uint32_t>((yy - top) * span[q] + (cx - left[q]) + 1);
                    tops[q].add(Match{cx, yy, 0, dist[q], ordinal});
                    if (filled[q] < keep) ++filled[q];
                    if (filled[q] == keep) worst[q] = out[static_cast<std::ptrdiff_t>(q) * os + keep].dist;
                }
            }
        }
    }
    if (hn::ReduceMax(du, bad_bits) >= 0x7f800000u) return -2;
    for (int i = 0; i < n; ++i) counts[i] = 1 + tops[i].finish();
    return 0;
}

#ifdef NSS_BM_KERNEL_LAB
nss::detail::Match8Kernel ResolveMatch8Lab(int v) {
#if HWY_MAX_BYTES >= 32
    switch (v) { case 0: return Match8LabEntry<0>; case 1: return Match8LabEntry<1>; case 2: return Match8LabEntry<2>; }
#endif
    return nullptr;
}
nss::detail::Match8Capture ResolveMatch8Capture(int v) {
#if HWY_MAX_BYTES >= 32
    switch (v) { case 0: return Match8CaptureEntry<0>; case 1: return Match8CaptureEntry<1>; case 2: return Match8CaptureEntry<2>; }
    return nullptr;
#else
    return nullptr;
#endif
}
nss::detail::Match8Replay ResolveMatch8Replay(int v) {
#if HWY_MAX_BYTES >= 32
    switch (v) { case 0: return Match8ReplayEntry<0>; case 1: return Match8ReplayEntry<1>; }
#endif
    return nullptr;
}
#endif

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
#ifdef NSS_BM_KERNEL_LAB
HWY_EXPORT(ResolveMatch8Lab);
HWY_EXPORT(ResolveMatch8Capture);
HWY_EXPORT(ResolveMatch8Replay);
namespace detail {
Match8Kernel match8_lab_kernel(int v) { return HWY_DYNAMIC_DISPATCH(ResolveMatch8Lab)(v); }
Match8Capture match8_capture_kernel(int v) { return HWY_DYNAMIC_DISPATCH(ResolveMatch8Capture)(v); }
Match8Replay match8_replay_kernel(int v) { return HWY_DYNAMIC_DISPATCH(ResolveMatch8Replay)(v); }
}
#endif
HWY_EXPORT(SsdBlock);
HWY_EXPORT(SsdNch);
HWY_EXPORT(SpatialMatch);
HWY_EXPORT(SpatialSquareRun8);

float ssd_block(const float* a, int sa, const float* b, int sb, int block) {
    return HWY_DYNAMIC_DISPATCH(SsdBlock)(a, sa, b, sb, block);
}

float ssd_nch(const float* const* a, const int* sa, const float* const* b, const int* sb, int nch, int block) {
    return HWY_DYNAMIC_DISPATCH(SsdNch)(a, sa, b, sb, nch, block);
}

int spatial_match(const float* ref, int stride, int width, int height, int bx, int by, int block, int bm_range,
                  int group, Match* out, unsigned avx2_features) {
    return HWY_DYNAMIC_DISPATCH(SpatialMatch)(ref, stride, width, height, bx, by, block, bm_range, group, out, avx2_features | NSS_AVX2_REQUESTED);
}

namespace detail {
int spatial_square_run8(const float* ref, int stride, int width, int height, int x0, int y, int n, int bm_range,
                        int group, Match* out, int os, int* counts) {
    return HWY_DYNAMIC_DISPATCH(SpatialSquareRun8)(ref, stride, width, height, x0, y, n, bm_range, group, out, os,
                                                   counts);
}
}  // namespace detail

}  // namespace nss
#endif
