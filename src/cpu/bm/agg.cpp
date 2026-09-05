#include "nss/cpu_api.hpp"
#include "cpu/hwy_config.hpp"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/bm/agg.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

#include <cstddef>

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

void AggFinish(float* dst, const float* num, const float* den, const float* src,
               int width, int height, int dstride, int buf_stride, int sstride) {
    const hn::ScalableTag<float> dtag;
    const int N = static_cast<int>(hn::Lanes(dtag));
    const auto eps = hn::Set(dtag, 1e-12f);
    if (sstride < 1) {
        sstride = dstride;
    }
    for (int y = 0; y < height; ++y) {
        int x = 0;
        float* o = dst + y * dstride;
        const float* n = num + y * buf_stride;
        const float* de = den + y * buf_stride;
        const float* s = src + y * sstride;
        for (; x + N <= width; x += N) {
            const auto dv = hn::LoadU(dtag, de + x);
            const auto nv = hn::LoadU(dtag, n + x);
            const auto sv = hn::LoadU(dtag, s + x);
            const auto ok = hn::Gt(dv, eps);
            hn::StoreU(hn::IfThenElse(ok, hn::Div(nv, dv), sv), dtag, o + x);
        }
        for (; x < width; ++x) {
            o[x] = (de[x] > 1e-12f) ? (n[x] / de[x]) : s[x];
        }
    }
}

template <int kSlices>
void VAggReduceImpl(float* dst, const float* fat, const float* src, int width, int height, int dstride,
                    int fstride, int sstride, int radius) {
    const int slices = kSlices > 0 ? kSlices : 2 * radius + 1;
    const hn::ScalableTag<float> dtag;
    const int N = static_cast<int>(hn::Lanes(dtag));
    const auto eps = hn::Set(dtag, 1e-12f);
    const std::ptrdiff_t plane_stride = static_cast<std::ptrdiff_t>(height) * fstride;
    const std::ptrdiff_t slice_stride = 2 * plane_stride;
    if (sstride < 1) {
        sstride = dstride;
    }
    for (int y = 0; y < height; ++y) {
        int x = 0;
        float* o = dst + y * dstride;
        const float* s = src + y * sstride;
        const float* num0 = fat + static_cast<std::ptrdiff_t>(y) * fstride;
        const float* den0 = num0 + plane_stride;
        for (; x + N <= width; x += N) {
            auto nv = hn::Zero(dtag);
            auto dv = hn::Zero(dtag);
            for (int t = 0; t < slices; ++t) {
                const std::ptrdiff_t offset = static_cast<std::ptrdiff_t>(t) * slice_stride + x;
                nv = hn::Add(nv, hn::LoadU(dtag, num0 + offset));
                dv = hn::Add(dv, hn::LoadU(dtag, den0 + offset));
            }
            const auto ok = hn::Gt(dv, eps);
            auto value = hn::Div(nv, dv);
            if (!hn::AllTrue(dtag, ok)) {
                value = hn::IfThenElse(ok, value, hn::LoadU(dtag, s + x));
            }
            hn::StoreU(value, dtag, o + x);
        }
        for (; x < width; ++x) {
            float num = 0.f;
            float den = 0.f;
            for (int t = 0; t < slices; ++t) {
                const std::ptrdiff_t offset = static_cast<std::ptrdiff_t>(t) * slice_stride + x;
                num += num0[offset];
                den += den0[offset];
            }
            o[x] = (den > 1e-12f) ? (num / den) : s[x];
        }
    }
}

void VAggReduceRuntime(float* dst, const float* fat, const float* src, int width, int height, int dstride,
                       int fstride, int sstride, int radius) {
    const int slices = 2 * radius + 1;
    const hn::ScalableTag<float> dtag;
    const int N = static_cast<int>(hn::Lanes(dtag));
    const auto eps = hn::Set(dtag, 1e-12f);
    if (sstride < 1) {
        sstride = dstride;
    }
    for (int y = 0; y < height; ++y) {
        int x = 0;
        float* o = dst + y * dstride;
        const float* s = src + y * sstride;
        for (; x + N <= width; x += N) {
            auto nv = hn::Zero(dtag);
            auto dv = hn::Zero(dtag);
            for (int t = 0; t < slices; ++t) {
                const int ynum = (t * 2) * height + y;
                const int yden = (t * 2 + 1) * height + y;
                nv = hn::Add(nv, hn::LoadU(dtag, fat + static_cast<std::ptrdiff_t>(ynum) * fstride + x));
                dv = hn::Add(dv, hn::LoadU(dtag, fat + static_cast<std::ptrdiff_t>(yden) * fstride + x));
            }
            hn::StoreU(hn::IfThenElse(hn::Gt(dv, eps), hn::Div(nv, dv), hn::LoadU(dtag, s + x)), dtag, o + x);
        }
        for (; x < width; ++x) {
            float num = 0.f;
            float den = 0.f;
            for (int t = 0; t < slices; ++t) {
                const int ynum = (t * 2) * height + y;
                const int yden = (t * 2 + 1) * height + y;
                num += fat[static_cast<std::ptrdiff_t>(ynum) * fstride + x];
                den += fat[static_cast<std::ptrdiff_t>(yden) * fstride + x];
            }
            o[x] = (den > 1e-12f) ? (num / den) : s[x];
        }
    }
}

void VAggReduce(float* dst, const float* fat, const float* src, int width, int height, int dstride,
                int fstride, int sstride, int radius) {
    switch (radius) {
        case 0:
            return VAggReduceImpl<1>(dst, fat, src, width, height, dstride, fstride, sstride, radius);
        case 1:
            return VAggReduceImpl<3>(dst, fat, src, width, height, dstride, fstride, sstride, radius);
        case 2:
            return VAggReduceImpl<5>(dst, fat, src, width, height, dstride, fstride, sstride, radius);
        case 3:
            return VAggReduceImpl<7>(dst, fat, src, width, height, dstride, fstride, sstride, radius);
        case 4:
            return VAggReduceImpl<9>(dst, fat, src, width, height, dstride, fstride, sstride, radius);
        default:
            return VAggReduceRuntime(dst, fat, src, width, height, dstride, fstride, sstride, radius);
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(AggFinish);
HWY_EXPORT(VAggReduce);

void aggregate_add(float* num, float* den, int stride, int x, int y,
                   const float* patch, int block, int width, int height, float w) {
    unpack_patch(num, den, stride, x, y, patch, block, width, height, w);
}

void aggregate_finish(float* dst, const float* num, const float* den, const float* src,
                      int width, int height, int dstride, int buf_stride, int sstride) {
    if (sstride < 1) {
        sstride = dstride;
    }
    HWY_DYNAMIC_DISPATCH(AggFinish)(dst, num, den, src, width, height, dstride, buf_stride, sstride);
}

void vaggregate_reduce(float* dst, const float* fat, const float* src,
                       int width, int height, int dstride, int fstride, int sstride, int radius) {
    HWY_DYNAMIC_DISPATCH(VAggReduce)(dst, fat, src, width, height, dstride, fstride, sstride, radius);
}

}  // namespace nss
#endif
