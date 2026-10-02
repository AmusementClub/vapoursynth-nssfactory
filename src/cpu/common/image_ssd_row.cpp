// SPDX-License-Identifier: GPL-2.0-only
#include "cpu/common/image_ssd_row.hpp"
#include "cpu/hwy_config.hpp"

#include <limits>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/common/image_ssd_row.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// The leaf functions are shared with cpu/bm/ssd.cpp (ssd_leaf-inl.hpp) so that
// every candidate distance is bit-identical to a ssd_block call on the same
// target: Ssd8 keeps the single 8-row accumulator + HSum8 tree, Ssd4 the 4-row chain,
// and other block sizes the scalable chunk + LoadN remainder + ReduceSum
// order. The row kernel only hoists dispatch, anchor loads, and candidate
// iteration; per-candidate expression trees are unchanged.

#include "cpu/bm/ssd_leaf-inl.hpp"

static float SsdBlockRow(const float* a, int sa, const float* b, int sb, int block) {
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

void ImageSsdRow(const float* anchor, int sa, const float* cand, int sb, int block, int count, float* out) {
    if (!anchor || !cand || !out || count < 1) {
        return;
    }
    if (block < 1 || sa < block || sb < block) {
        // Same finite sentinel as ssd_block for invalid views.
        for (int x = 0; x < count; ++x) {
            out[x] = std::numeric_limits<float>::max();
        }
        return;
    }
#if HWY_MAX_BYTES >= 32
    if (block == 8) {
        const hn::FixedTag<float, 8> df;
        hn::Vec<decltype(df)> refb[8];
        for (int i = 0; i < 8; ++i) {
            refb[i] = hn::LoadU(df, anchor + i * sa);
        }
        int x = 0;
        // Four candidates in flight, each with ONE accumulator chained over
        // the same 8 rows as Ssd8; interleaving independent chains does not
        // change any candidate's value.
        for (; x + 4 <= count; x += 4) {
            auto acc0 = hn::Zero(df);
            auto acc1 = hn::Zero(df);
            auto acc2 = hn::Zero(df);
            auto acc3 = hn::Zero(df);
            const float* row = cand + x;
            for (int i = 0; i < 8; ++i) {
                const auto r = refb[i];
                const auto d0 = hn::Sub(r, hn::LoadU(df, row + 0 + i * sb));
                const auto d1 = hn::Sub(r, hn::LoadU(df, row + 1 + i * sb));
                const auto d2 = hn::Sub(r, hn::LoadU(df, row + 2 + i * sb));
                const auto d3 = hn::Sub(r, hn::LoadU(df, row + 3 + i * sb));
                acc0 = hn::MulAdd(d0, d0, acc0);
                acc1 = hn::MulAdd(d1, d1, acc1);
                acc2 = hn::MulAdd(d2, d2, acc2);
                acc3 = hn::MulAdd(d3, d3, acc3);
            }
            out[x + 0] = HSum8(acc0);
            out[x + 1] = HSum8(acc1);
            out[x + 2] = HSum8(acc2);
            out[x + 3] = HSum8(acc3);
        }
        for (; x < count; ++x) {
            out[x] = Ssd8(anchor, sa, cand + x, sb);
        }
        return;
    }
#endif
    for (int x = 0; x < count; ++x) {
        out[x] = SsdBlockRow(anchor, sa, cand + x, sb, block);
    }
}

ImageSsdRowKernel ResolveImageSsdRow() { return &ImageSsdRow; }

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(ResolveImageSsdRow);

ImageSsdRowKernel image_ssd_row_kernel() { return HWY_DYNAMIC_DISPATCH(ResolveImageSsdRow)(); }

}  // namespace nss
#endif
