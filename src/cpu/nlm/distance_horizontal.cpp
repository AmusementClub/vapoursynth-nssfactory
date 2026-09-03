#include "nss/cpu_api.hpp"
#include "cpu/hwy_config.hpp"

#include <algorithm>
#include <cmath>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/nlm/distance_horizontal.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

static inline int clampi(int value, int lo, int hi) {
    return value < lo ? lo : (value > hi ? hi : value);
}

static inline void distance_luma_row(float* dst, const float* center, const float* neighbor, int ox, int oy,
                                     int y, int w, int h, int stride) {
    const int start_x = std::abs(ox);
    const int end_x = w - start_x;
    const int neighbor_y = clampi(y + oy, 0, h - 1);
    const hn::ScalableTag<float> d;
    const int lanes = static_cast<int>(hn::Lanes(d));
    const auto three = hn::Set(d, 3.f);
    int x = 0;
    for (; x < start_x; ++x) {
        const float delta = center[y * stride + x] -
                            neighbor[neighbor_y * stride + clampi(x + ox, 0, w - 1)];
        dst[x] = 3.f * delta * delta;
    }
    for (; x + lanes <= end_x; x += lanes) {
        const auto delta = hn::Sub(hn::LoadU(d, center + y * stride + x),
                                   hn::LoadU(d, neighbor + neighbor_y * stride + x + ox));
        hn::StoreU(hn::Mul(hn::Mul(delta, delta), three), d, dst + x);
    }
    for (; x < end_x; ++x) {
        const float delta = center[y * stride + x] - neighbor[neighbor_y * stride + x + ox];
        dst[x] = 3.f * delta * delta;
    }
    for (; x < w; ++x) {
        const float delta = center[y * stride + x] -
                            neighbor[neighbor_y * stride + clampi(x + ox, 0, w - 1)];
        dst[x] = 3.f * delta * delta;
    }
}

template <int Radius>
static inline void horizontal_fixed_row(float* dst, const float* src, int w) {
    const hn::ScalableTag<float> d;
    const int lanes = static_cast<int>(hn::Lanes(d));
    int x = 0;
    for (; x < Radius && x < w; ++x) {
        float sum = 0.f;
        for (int j = -Radius; j <= Radius; ++j) {
            sum += src[clampi(x + j, 0, w - 1)];
        }
        dst[x] = sum;
    }
    const int inner_end = w - Radius;
    for (; x + lanes <= inner_end; x += lanes) {
        const auto a0 = hn::LoadU(d, src + x - Radius);
        const auto a1 = hn::LoadU(d, src + x - Radius + 1);
        const auto a2 = hn::LoadU(d, src + x - Radius + 2);
        if constexpr (Radius == 1) {
            hn::StoreU(hn::Add(hn::Add(a0, a1), a2), d, dst + x);
        } else {
            const auto a3 = hn::LoadU(d, src + x - Radius + 3);
            const auto a4 = hn::LoadU(d, src + x - Radius + 4);
            const auto p01 = hn::Add(a0, a1);
            const auto p23 = hn::Add(a2, a3);
            if constexpr (Radius == 2) {
                hn::StoreU(hn::Add(hn::Add(p01, p23), a4), d, dst + x);
            } else {
                const auto a5 = hn::LoadU(d, src + x - Radius + 5);
                const auto a6 = hn::LoadU(d, src + x - Radius + 6);
                const auto p45 = hn::Add(a4, a5);
                if constexpr (Radius == 3) {
                    hn::StoreU(hn::Add(hn::Add(p01, p23), hn::Add(p45, a6)), d, dst + x);
                } else {
                    const auto a7 = hn::LoadU(d, src + x - Radius + 7);
                    const auto a8 = hn::LoadU(d, src + x - Radius + 8);
                    const auto p67 = hn::Add(a6, a7);
                    hn::StoreU(hn::Add(hn::Add(p01, p23), hn::Add(hn::Add(p45, p67), a8)), d, dst + x);
                }
            }
        }
    }
    for (; x < inner_end; ++x) {
        float sum = 0.f;
        for (int j = -Radius; j <= Radius; ++j) {
            sum += src[x + j];
        }
        dst[x] = sum;
    }
    for (; x < w; ++x) {
        float sum = 0.f;
        for (int j = -Radius; j <= Radius; ++j) {
            sum += src[clampi(x + j, 0, w - 1)];
        }
        dst[x] = sum;
    }
}

static inline void horizontal_row(float* dst, const float* src, int radius, int w) {
    const hn::ScalableTag<float> d;
    const int lanes = static_cast<int>(hn::Lanes(d));
    const int left = std::min(radius, w);
    int x = 0;
    for (; x < left; ++x) {
        float sum = 0.f;
        for (int j = -radius; j <= radius; ++j) {
            sum += src[clampi(x + j, 0, w - 1)];
        }
        dst[x] = sum;
    }
    const int inner_end = w - radius;
    if (inner_end > left) {
        for (; x + lanes <= inner_end; x += lanes) {
            auto sum = hn::Zero(d);
            for (int j = -radius; j <= radius; ++j) {
                sum = hn::Add(sum, hn::LoadU(d, src + x + j));
            }
            hn::StoreU(sum, d, dst + x);
        }
        for (; x < inner_end; ++x) {
            float sum = 0.f;
            for (int j = -radius; j <= radius; ++j) {
                sum += src[x + j];
            }
            dst[x] = sum;
        }
    }
    for (; x < w; ++x) {
        float sum = 0.f;
        for (int j = -radius; j <= radius; ++j) {
            sum += src[clampi(x + j, 0, w - 1)];
        }
        dst[x] = sum;
    }
}

template <int Radius>
static void distance_horizontal_fixed(float* dst, float* row_scratch, const float* center,
                                      const float* neighbor, int ox, int oy, int w, int h, int stride) {
    for (int y = 0; y < h; ++y) {
        distance_luma_row(row_scratch, center, neighbor, ox, oy, y, w, h, stride);
        horizontal_fixed_row<Radius>(dst + y * stride, row_scratch, w);
    }
}

void DistanceLumaHorizontal(float* dst, float* row_scratch, const float* center, const float* neighbor, int ox,
                            int oy, int radius, int w, int h, int stride) {
    if (!dst || !row_scratch || !center || !neighbor || radius < 0 || w < 1 || h < 1 || stride < w) {
        return;
    }
    if (radius == 1) {
        distance_horizontal_fixed<1>(dst, row_scratch, center, neighbor, ox, oy, w, h, stride);
    } else if (radius == 2) {
        distance_horizontal_fixed<2>(dst, row_scratch, center, neighbor, ox, oy, w, h, stride);
    } else if (radius == 3) {
        distance_horizontal_fixed<3>(dst, row_scratch, center, neighbor, ox, oy, w, h, stride);
    } else if (radius == 4) {
        distance_horizontal_fixed<4>(dst, row_scratch, center, neighbor, ox, oy, w, h, stride);
    } else {
        for (int y = 0; y < h; ++y) {
            distance_luma_row(row_scratch, center, neighbor, ox, oy, y, w, h, stride);
            horizontal_row(dst + y * stride, row_scratch, radius, w);
        }
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(DistanceLumaHorizontal);

void nlm_distance_luma_horizontal_f32(float* dst, float* row_scratch, const float* center,
                                      const float* neighbor, int ox, int oy, int radius,
                                      int w, int h, int stride) {
    HWY_DYNAMIC_DISPATCH(DistanceLumaHorizontal)(dst, row_scratch, center, neighbor, ox, oy, radius, w, h, stride);
}
}  // namespace nss
#endif
