#include "nss/cpu_api.hpp"
#include "cpu/hwy_config.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/nlm/nlm.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include "hwy/contrib/math/fast_math-inl.h"
#include "hwy/contrib/math/math-inl.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

static inline int clampi(int x, int lo, int hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static inline void add_row(float* dst, const float* src, int n) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    int x = 0;
    for (; x + N <= n; x += N) {
        hn::StoreU(hn::Add(hn::LoadU(d, dst + x), hn::LoadU(d, src + x)), d, dst + x);
    }
    for (; x < n; ++x) {
        dst[x] += src[x];
    }
}

static inline void add_exp_sub_row(float* dst, float* buffer, const float* add, const float* sub, float scale, int n) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const auto vs = hn::Set(d, scale);
    int x = 0;
    for (; x + N <= n; x += N) {
        const auto sum = hn::Add(hn::LoadU(d, buffer + x), hn::LoadU(d, add + x));
        const auto exponent = hn::Mul(sum, vs);
        hn::StoreU(hn::FastExp</*kHandleSubnormals=*/false>(d, exponent), d, dst + x);
        hn::StoreU(hn::Sub(sum, hn::LoadU(d, sub + x)), d, buffer + x);
    }
    if (x < n) {
        const std::size_t remaining = static_cast<std::size_t>(n - x);
        const auto sum = hn::Add(hn::LoadN(d, buffer + x, remaining), hn::LoadN(d, add + x, remaining));
        hn::StoreN(hn::FastExp</*kHandleSubnormals=*/false>(d, hn::Mul(sum, vs)), d, dst + x, remaining);
        hn::StoreN(hn::Sub(sum, hn::LoadN(d, sub + x, remaining)), d, buffer + x, remaining);
    }
}

static inline void add_sub_row(float* buffer, const float* add, const float* sub, int n) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    int x = 0;
    for (; x + N <= n; x += N) {
        const auto sum = hn::Add(hn::LoadU(d, buffer + x), hn::LoadU(d, add + x));
        hn::StoreU(hn::Sub(sum, hn::LoadU(d, sub + x)), d, buffer + x);
    }
    for (; x < n; ++x) {
        buffer[x] = (buffer[x] + add[x]) - sub[x];
    }
}

static inline void scale_row(float* dst, const float* src, float s, int n) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const auto vs = hn::Set(d, s);
    int x = 0;
    for (; x + N <= n; x += N) {
        hn::StoreU(hn::Mul(vs, hn::LoadU(d, src + x)), d, dst + x);
    }
    for (; x < n; ++x) {
        dst[x] = s * src[x];
    }
}

static inline void store_sq3(float* dst, const float* a, const float* b, int n) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const auto three = hn::Set(d, 3.0f);
    int x = 0;
    for (; x + N <= n; x += N) {
        const auto da = hn::Sub(hn::LoadU(d, a + x), hn::LoadU(d, b + x));
        hn::StoreU(hn::Mul(hn::Mul(da, da), three), d, dst + x);
    }
    for (; x < n; ++x) {
        const float t = a[x] - b[x];
        dst[x] = 3.0f * t * t;
    }
}

static inline void store_chroma(float* dst, const float* c1, const float* c2, const float* n1, const float* n2,
                                int n) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const auto one_half = hn::Set(d, 1.5f);
    int x = 0;
    for (; x + N <= n; x += N) {
        const auto d1 = hn::Sub(hn::LoadU(d, c1 + x), hn::LoadU(d, n1 + x));
        const auto d2 = hn::Sub(hn::LoadU(d, c2 + x), hn::LoadU(d, n2 + x));
        hn::StoreU(hn::Mul(one_half, hn::MulAdd(d1, d1, hn::Mul(d2, d2))), d, dst + x);
    }
    for (; x < n; ++x) {
        const float e1 = c1[x] - n1[x];
        const float e2 = c2[x] - n2[x];
        dst[x] = 1.5f * (e1 * e1 + e2 * e2);
    }
}

static inline void store_yuv(float* dst, const float* c0, const float* c1, const float* c2, const float* n0,
                             const float* n1, const float* n2, int n) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    int x = 0;
    for (; x + N <= n; x += N) {
        const auto e0 = hn::Sub(hn::LoadU(d, c0 + x), hn::LoadU(d, n0 + x));
        const auto e1 = hn::Sub(hn::LoadU(d, c1 + x), hn::LoadU(d, n1 + x));
        const auto e2 = hn::Sub(hn::LoadU(d, c2 + x), hn::LoadU(d, n2 + x));
        hn::StoreU(hn::MulAdd(e2, e2, hn::MulAdd(e1, e1, hn::Mul(e0, e0))), d, dst + x);
    }
    for (; x < n; ++x) {
        const float e0 = c0[x] - n0[x];
        const float e1 = c1[x] - n1[x];
        const float e2 = c2[x] - n2[x];
        dst[x] = e0 * e0 + e1 * e1 + e2 * e2;
    }
}

static inline void store_rgb(float* dst, const float* c0, const float* c1, const float* c2, const float* n0,
                             const float* n1, const float* n2, int n) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const auto sixth = hn::Set(d, 1.0f / 6.0f);
    const auto two_third = hn::Set(d, 2.0f / 3.0f);
    const auto four_third = hn::Set(d, 4.0f / 3.0f);
    const auto one = hn::Set(d, 1.0f);
    int x = 0;
    for (; x + N <= n; x += N) {
        const auto u1 = hn::LoadU(d, c0 + x);
        const auto v1 = hn::LoadU(d, n0 + x);
        const auto u2 = hn::LoadU(d, c1 + x);
        const auto v2 = hn::LoadU(d, n1 + x);
        const auto u3 = hn::LoadU(d, c2 + x);
        const auto v3 = hn::LoadU(d, n2 + x);
        const auto m_red = hn::Mul(hn::Add(u1, v1), sixth);
        const auto d1 = hn::Sub(u1, v1);
        const auto d2 = hn::Sub(u2, v2);
        const auto d3 = hn::Sub(u3, v3);
        auto acc = hn::Mul(hn::Add(two_third, m_red), hn::Mul(d1, d1));
        acc = hn::MulAdd(four_third, hn::Mul(d2, d2), acc);
        acc = hn::MulAdd(hn::Sub(one, m_red), hn::Mul(d3, d3), acc);
        hn::StoreU(acc, d, dst + x);
    }
    for (; x < n; ++x) {
        const float u1 = c0[x], v1 = n0[x];
        const float u2 = c1[x], v2 = n1[x];
        const float u3 = c2[x], v3 = n2[x];
        const float m_red = (u1 + v1) / 6.0f;
        dst[x] = (2.0f / 3.0f + m_red) * (u1 - v1) * (u1 - v1) + (4.0f / 3.0f) * (u2 - v2) * (u2 - v2) +
                 (1.0f - m_red) * (u3 - v3) * (u3 - v3);
    }
}

void DistanceLuma(float* dst, const float* center, const float* neighbor, int ox, int oy, int w, int h,
                  int stride) {
    const int start_x = std::abs(ox);
    const int end_x = w - std::abs(ox);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < start_x; ++x) {
            const int idx = y * stride + x;
            const int ni = clampi(y + oy, 0, h - 1) * stride + clampi(x + ox, 0, w - 1);
            const float t = center[idx] - neighbor[ni];
            dst[idx] = 3.0f * t * t;
        }
        if (end_x > start_x) {
            const int y2 = clampi(y + oy, 0, h - 1);
            store_sq3(dst + y * stride + start_x, center + y * stride + start_x,
                      neighbor + y2 * stride + start_x + ox, end_x - start_x);
        }
        for (int x = std::max(end_x, start_x); x < w; ++x) {
            const int idx = y * stride + x;
            const int ni = clampi(y + oy, 0, h - 1) * stride + clampi(x + ox, 0, w - 1);
            const float t = center[idx] - neighbor[ni];
            dst[idx] = 3.0f * t * t;
        }
    }
}

void DistanceChroma(float* dst, const float* c1, const float* c2, const float* n1, const float* n2, int ox,
                    int oy, int w, int h, int stride) {
    const int start_x = std::abs(ox);
    const int end_x = w - std::abs(ox);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < start_x; ++x) {
            const int idx = y * stride + x;
            const int ni = clampi(y + oy, 0, h - 1) * stride + clampi(x + ox, 0, w - 1);
            const float d1 = c1[idx] - n1[ni];
            const float d2 = c2[idx] - n2[ni];
            dst[idx] = 1.5f * (d1 * d1 + d2 * d2);
        }
        if (end_x > start_x) {
            const int y2 = clampi(y + oy, 0, h - 1);
            const int off = y * stride + start_x;
            const int noff = y2 * stride + start_x + ox;
            store_chroma(dst + off, c1 + off, c2 + off, n1 + noff, n2 + noff, end_x - start_x);
        }
        for (int x = std::max(end_x, start_x); x < w; ++x) {
            const int idx = y * stride + x;
            const int ni = clampi(y + oy, 0, h - 1) * stride + clampi(x + ox, 0, w - 1);
            const float d1 = c1[idx] - n1[ni];
            const float d2 = c2[idx] - n2[ni];
            dst[idx] = 1.5f * (d1 * d1 + d2 * d2);
        }
    }
}

void DistanceYUV(float* dst, const float* c0, const float* c1, const float* c2, const float* n0, const float* n1,
                 const float* n2, int ox, int oy, int w, int h, int stride) {
    const int start_x = std::abs(ox);
    const int end_x = w - std::abs(ox);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < start_x; ++x) {
            const int idx = y * stride + x;
            const int ni = clampi(y + oy, 0, h - 1) * stride + clampi(x + ox, 0, w - 1);
            const float d0 = c0[idx] - n0[ni];
            const float d1 = c1[idx] - n1[ni];
            const float d2 = c2[idx] - n2[ni];
            dst[idx] = d0 * d0 + d1 * d1 + d2 * d2;
        }
        if (end_x > start_x) {
            const int y2 = clampi(y + oy, 0, h - 1);
            const int off = y * stride + start_x;
            const int noff = y2 * stride + start_x + ox;
            store_yuv(dst + off, c0 + off, c1 + off, c2 + off, n0 + noff, n1 + noff, n2 + noff, end_x - start_x);
        }
        for (int x = std::max(end_x, start_x); x < w; ++x) {
            const int idx = y * stride + x;
            const int ni = clampi(y + oy, 0, h - 1) * stride + clampi(x + ox, 0, w - 1);
            const float d0 = c0[idx] - n0[ni];
            const float d1 = c1[idx] - n1[ni];
            const float d2 = c2[idx] - n2[ni];
            dst[idx] = d0 * d0 + d1 * d1 + d2 * d2;
        }
    }
}

void DistanceRGB(float* dst, const float* c0, const float* c1, const float* c2, const float* n0, const float* n1,
                 const float* n2, int ox, int oy, int w, int h, int stride) {
    const int start_x = std::abs(ox);
    const int end_x = w - std::abs(ox);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < start_x; ++x) {
            const int idx = y * stride + x;
            const int ni = clampi(y + oy, 0, h - 1) * stride + clampi(x + ox, 0, w - 1);
            const float u1 = c0[idx], v1 = n0[ni];
            const float u2 = c1[idx], v2 = n1[ni];
            const float u3 = c2[idx], v3 = n2[ni];
            const float m_red = (u1 + v1) / 6.0f;
            dst[idx] = (2.0f / 3.0f + m_red) * (u1 - v1) * (u1 - v1) + (4.0f / 3.0f) * (u2 - v2) * (u2 - v2) +
                       (1.0f - m_red) * (u3 - v3) * (u3 - v3);
        }
        if (end_x > start_x) {
            const int y2 = clampi(y + oy, 0, h - 1);
            const int off = y * stride + start_x;
            const int noff = y2 * stride + start_x + ox;
            store_rgb(dst + off, c0 + off, c1 + off, c2 + off, n0 + noff, n1 + noff, n2 + noff, end_x - start_x);
        }
        for (int x = std::max(end_x, start_x); x < w; ++x) {
            const int idx = y * stride + x;
            const int ni = clampi(y + oy, 0, h - 1) * stride + clampi(x + ox, 0, w - 1);
            const float u1 = c0[idx], v1 = n0[ni];
            const float u2 = c1[idx], v2 = n1[ni];
            const float u3 = c2[idx], v3 = n2[ni];
            const float m_red = (u1 + v1) / 6.0f;
            dst[idx] = (2.0f / 3.0f + m_red) * (u1 - v1) * (u1 - v1) + (4.0f / 3.0f) * (u2 - v2) * (u2 - v2) +
                       (1.0f - m_red) * (u3 - v3) * (u3 - v3);
        }
    }
}

template <int S>
static void HorizontalFixedS(float* dst, const float* src, int w, int h, int stride) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    for (int y = 0; y < h; ++y) {
        const float* row = src + y * stride;
        float* out = dst + y * stride;
        for (int x = 0; x < S && x < w; ++x) {
            float sum = 0.f;
            for (int j = -S; j <= S; ++j) {
                sum += row[clampi(x + j, 0, w - 1)];
            }
            out[x] = sum;
        }
        const int inner_end = w - S;
        int x = S;
        for (; x + N <= inner_end; x += N) {
            const auto acc = [&] {
                const auto a0 = hn::LoadU(d, row + x - S);
                const auto a1 = hn::LoadU(d, row + x - S + 1);
                const auto a2 = hn::LoadU(d, row + x - S + 2);
                if constexpr (S == 1) {
                    return hn::Add(hn::Add(a0, a1), a2);
                }
                const auto a3 = hn::LoadU(d, row + x - S + 3);
                const auto a4 = hn::LoadU(d, row + x - S + 4);
                const auto p01 = hn::Add(a0, a1);
                const auto p23 = hn::Add(a2, a3);
                if constexpr (S == 2) {
                    return hn::Add(hn::Add(p01, p23), a4);
                }
                const auto a5 = hn::LoadU(d, row + x - S + 5);
                const auto a6 = hn::LoadU(d, row + x - S + 6);
                const auto p45 = hn::Add(a4, a5);
                if constexpr (S == 3) {
                    return hn::Add(hn::Add(p01, p23), hn::Add(p45, a6));
                }
                const auto a7 = hn::LoadU(d, row + x - S + 7);
                const auto a8 = hn::LoadU(d, row + x - S + 8);
                const auto p67 = hn::Add(a6, a7);
                return hn::Add(hn::Add(p01, p23), hn::Add(hn::Add(p45, p67), a8));
            }();
            hn::StoreU(acc, d, out + x);
        }
        for (; x < inner_end; ++x) {
            float sum = 0.f;
            for (int j = -S; j <= S; ++j) {
                sum += row[x + j];
            }
            out[x] = sum;
        }
        for (; x < w; ++x) {
            float sum = 0.f;
            for (int j = -S; j <= S; ++j) {
                sum += row[clampi(x + j, 0, w - 1)];
            }
            out[x] = sum;
        }
    }
}

void Horizontal(float* dst, const float* src, int s, int w, int h, int stride) {
    if (s == 1) {
        HorizontalFixedS<1>(dst, src, w, h, stride);
        return;
    }
    if (s == 2) {
        HorizontalFixedS<2>(dst, src, w, h, stride);
        return;
    }
    if (s == 3) {
        HorizontalFixedS<3>(dst, src, w, h, stride);
        return;
    }
    if (s == 4) {
        HorizontalFixedS<4>(dst, src, w, h, stride);
        return;
    }
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    for (int y = 0; y < h; ++y) {
        const float* row = src + y * stride;
        float* out = dst + y * stride;
        const int left = std::min(s, w);
        for (int x = 0; x < left; ++x) {
            float sum = 0.f;
            for (int j = -s; j <= s; ++j) {
                sum += row[clampi(x + j, 0, w - 1)];
            }
            out[x] = sum;
        }
        const int inner_end = w - s;
        int x = left;
        if (inner_end > left) {
            for (; x + N <= inner_end; x += N) {
                auto acc = hn::Zero(d);
                for (int j = -s; j <= s; ++j) {
                    acc = hn::Add(acc, hn::LoadU(d, row + x + j));
                }
                hn::StoreU(acc, d, out + x);
            }
            for (; x < inner_end; ++x) {
                float sum = 0.f;
                for (int j = -s; j <= s; ++j) {
                    sum += row[x + j];
                }
                out[x] = sum;
            }
        }
        for (; x < w; ++x) {
            float sum = 0.f;
            for (int j = -s; j <= s; ++j) {
                sum += row[clampi(x + j, 0, w - 1)];
            }
            out[x] = sum;
        }
    }
}

void VerticalWelschRange(float* dst, const float* src, int radius, float h2_inv_norm, int w, int h, int stride,
                         int y0, int y1, float* buffer) {
    const float nscale = -h2_inv_norm;
    scale_row(buffer, src, static_cast<float>(radius), w);
    for (int y = 0; y < radius; ++y) {
        const int yy = std::min(y, h - 1);
        add_row(buffer, src + yy * stride, w);
    }
    for (int y = 0; y < y1; ++y) {
        const int yadd = std::min(y + radius, h - 1);
        const int ysub = std::max(y - radius, 0);
        if (y < y0) {
            add_sub_row(buffer, src + yadd * stride, src + ysub * stride, w);
        } else {
            add_exp_sub_row(dst + (y - y0) * stride, buffer, src + yadd * stride, src + ysub * stride, nscale, w);
        }
    }
}

void VerticalWelsch(float* dst, const float* src, int radius, float h2_inv_norm, int w, int h, int stride,
                    float* buffer) {
    VerticalWelschRange(dst, src, radius, h2_inv_norm, w, h, stride, 0, h, buffer);
}

static void accum_inner(float* weight, float* wdst0, float* wdst1, float* wdst2, float* maxw,
                        const float* s0_bwd, const float* s1_bwd, const float* s2_bwd, const float* s0_fwd,
                        const float* s1_fwd, const float* s2_fwd, const float* temp1, const float* temp2,
                        int ox, int oy, int w, int h, int stride, int nch, int y0, int y1, int temp1_base_y,
                        int temp2_base_y) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const int start_x = std::abs(ox);
    const int end_x = w - std::abs(ox);
    for (int y = y0; y < y1; ++y) {
        const int out_y = y - y0;
        const int temp1_y = y - temp1_base_y;
        for (int x = 0; x < start_x; ++x) {
            const int idx = temp1_y * stride + x;
            const int out_idx = out_y * stride + x;
            const float u4 = temp1[idx];
            const int mq_y = clampi(y - oy, 0, h - 1);
            const int mq_x = clampi(x - ox, 0, w - 1);
            const int mq = mq_y * stride + mq_x;
            const int temp2_mq = (mq_y - temp2_base_y) * stride + mq_x;
            const float u4_mq = temp2[temp2_mq];
            weight[out_idx] += u4 + u4_mq;
            maxw[out_idx] = std::max(std::max(u4, u4_mq), maxw[out_idx]);
            const int pq = clampi(y + oy, 0, h - 1) * stride + clampi(x + ox, 0, w - 1);
            wdst0[out_idx] += u4 * s0_bwd[pq] + u4_mq * s0_fwd[mq];
            if (nch > 1) {
                wdst1[out_idx] += u4 * s1_bwd[pq] + u4_mq * s1_fwd[mq];
            }
            if (nch > 2) {
                wdst2[out_idx] += u4 * s2_bwd[pq] + u4_mq * s2_fwd[mq];
            }
        }
        if (end_x > start_x) {
            const int ypq = clampi(y + oy, 0, h - 1);
            const int ymq = clampi(y - oy, 0, h - 1);
            const int off = temp1_y * stride + start_x;
            const int out_off = out_y * stride + start_x;
            const int mq0 = ymq * stride + start_x - ox;
            const int temp2_mq0 = (ymq - temp2_base_y) * stride + start_x - ox;
            const int pq0 = ypq * stride + start_x + ox;
            const int n = end_x - start_x;
            int x = 0;
            for (; x + N <= n; x += N) {
                const auto u4 = hn::LoadU(d, temp1 + off + x);
                const auto u4m = hn::LoadU(d, temp2 + temp2_mq0 + x);
                const auto zero = hn::Zero(d);
                if (hn::AllTrue(d, hn::And(hn::Eq(u4, zero), hn::Eq(u4m, zero)))) {
                    continue;
                }
                hn::StoreU(hn::Add(hn::LoadU(d, weight + out_off + x), hn::Add(u4, u4m)), d, weight + out_off + x);
                hn::StoreU(hn::Max(hn::Max(u4, u4m), hn::LoadU(d, maxw + out_off + x)), d, maxw + out_off + x);
                auto acc0 = hn::MulAdd(u4, hn::LoadU(d, s0_bwd + pq0 + x), hn::Mul(u4m, hn::LoadU(d, s0_fwd + mq0 + x)));
                hn::StoreU(hn::Add(hn::LoadU(d, wdst0 + out_off + x), acc0), d, wdst0 + out_off + x);
                if (nch > 1) {
                    auto acc1 =
                        hn::MulAdd(u4, hn::LoadU(d, s1_bwd + pq0 + x), hn::Mul(u4m, hn::LoadU(d, s1_fwd + mq0 + x)));
                    hn::StoreU(hn::Add(hn::LoadU(d, wdst1 + out_off + x), acc1), d, wdst1 + out_off + x);
                }
                if (nch > 2) {
                    auto acc2 =
                        hn::MulAdd(u4, hn::LoadU(d, s2_bwd + pq0 + x), hn::Mul(u4m, hn::LoadU(d, s2_fwd + mq0 + x)));
                    hn::StoreU(hn::Add(hn::LoadU(d, wdst2 + out_off + x), acc2), d, wdst2 + out_off + x);
                }
            }
            for (; x < n; ++x) {
                const float u4 = temp1[off + x];
                const float u4_mq = temp2[temp2_mq0 + x];
                weight[out_off + x] += u4 + u4_mq;
                maxw[out_off + x] = std::max(std::max(u4, u4_mq), maxw[out_off + x]);
                wdst0[out_off + x] += u4 * s0_bwd[pq0 + x] + u4_mq * s0_fwd[mq0 + x];
                if (nch > 1) {
                    wdst1[out_off + x] += u4 * s1_bwd[pq0 + x] + u4_mq * s1_fwd[mq0 + x];
                }
                if (nch > 2) {
                    wdst2[out_off + x] += u4 * s2_bwd[pq0 + x] + u4_mq * s2_fwd[mq0 + x];
                }
            }
        }
        for (int x = std::max(end_x, start_x); x < w; ++x) {
            const int idx = temp1_y * stride + x;
            const int out_idx = out_y * stride + x;
            const float u4 = temp1[idx];
            const int mq_y = clampi(y - oy, 0, h - 1);
            const int mq_x = clampi(x - ox, 0, w - 1);
            const int mq = mq_y * stride + mq_x;
            const int temp2_mq = (mq_y - temp2_base_y) * stride + mq_x;
            const float u4_mq = temp2[temp2_mq];
            weight[out_idx] += u4 + u4_mq;
            maxw[out_idx] = std::max(std::max(u4, u4_mq), maxw[out_idx]);
            const int pq = clampi(y + oy, 0, h - 1) * stride + clampi(x + ox, 0, w - 1);
            wdst0[out_idx] += u4 * s0_bwd[pq] + u4_mq * s0_fwd[mq];
            if (nch > 1) {
                wdst1[out_idx] += u4 * s1_bwd[pq] + u4_mq * s1_fwd[mq];
            }
            if (nch > 2) {
                wdst2[out_idx] += u4 * s2_bwd[pq] + u4_mq * s2_fwd[mq];
            }
        }
    }
}

void AccumCh1(float* weight, float* wdst, float* maxw, const float* src_bwd, const float* src_fwd,
              const float* temp1, const float* temp2, int ox, int oy, int w, int h, int stride) {
    accum_inner(weight, wdst, nullptr, nullptr, maxw, src_bwd, nullptr, nullptr, src_fwd, nullptr, nullptr, temp1,
                temp2, ox, oy, w, h, stride, 1, 0, h, 0, 0);
}

void AccumCh2(float* weight, float* wdst0, float* wdst1, float* maxw, const float* s0_bwd, const float* s1_bwd,
              const float* s0_fwd, const float* s1_fwd, const float* temp1, const float* temp2, int ox, int oy, int w,
              int h, int stride) {
    accum_inner(weight, wdst0, wdst1, nullptr, maxw, s0_bwd, s1_bwd, nullptr, s0_fwd, s1_fwd, nullptr, temp1, temp2, ox,
                oy, w, h, stride, 2, 0, h, 0, 0);
}

void AccumCh3(float* weight, float* wdst0, float* wdst1, float* wdst2, float* maxw, const float* s0_bwd,
              const float* s1_bwd, const float* s2_bwd, const float* s0_fwd, const float* s1_fwd, const float* s2_fwd,
              const float* temp1, const float* temp2, int ox, int oy, int w, int h, int stride) {
    accum_inner(weight, wdst0, wdst1, wdst2, maxw, s0_bwd, s1_bwd, s2_bwd, s0_fwd, s1_fwd, s2_fwd, temp1, temp2, ox, oy,
                w, h, stride, 3, 0, h, 0, 0);
}

void AccumCh1Range(float* weight, float* wdst, float* maxw, const float* src_bwd, const float* src_fwd,
                   const float* temp1, const float* temp2, int ox, int oy, int w, int h, int stride, int y0, int y1) {
    accum_inner(weight, wdst, nullptr, nullptr, maxw, src_bwd, nullptr, nullptr, src_fwd, nullptr, nullptr, temp1,
                temp2, ox, oy, w, h, stride, 1, y0, y1, 0, 0);
}

void AccumCh2Range(float* weight, float* wdst0, float* wdst1, float* maxw, const float* s0_bwd, const float* s1_bwd,
                   const float* s0_fwd, const float* s1_fwd, const float* temp1, const float* temp2, int ox, int oy,
                   int w, int h, int stride, int y0, int y1) {
    accum_inner(weight, wdst0, wdst1, nullptr, maxw, s0_bwd, s1_bwd, nullptr, s0_fwd, s1_fwd, nullptr, temp1, temp2,
                ox, oy, w, h, stride, 2, y0, y1, 0, 0);
}

void AccumCh3Range(float* weight, float* wdst0, float* wdst1, float* wdst2, float* maxw, const float* s0_bwd,
                   const float* s1_bwd, const float* s2_bwd, const float* s0_fwd, const float* s1_fwd,
                   const float* s2_fwd, const float* temp1, const float* temp2, int ox, int oy, int w, int h, int stride,
                   int y0, int y1) {
    accum_inner(weight, wdst0, wdst1, wdst2, maxw, s0_bwd, s1_bwd, s2_bwd, s0_fwd, s1_fwd, s2_fwd, temp1, temp2, ox,
                oy, w, h, stride, 3, y0, y1, 0, 0);
}

void AccumCh1CoreRange(float* weight, float* wdst, float* maxw, const float* src_bwd, const float* src_fwd,
                       const float* temp1_core, const float* temp2, int ox, int oy, int w, int h, int stride, int y0,
                       int y1, int temp2_base_y) {
    accum_inner(weight, wdst, nullptr, nullptr, maxw, src_bwd, nullptr, nullptr, src_fwd, nullptr, nullptr, temp1_core,
                temp2, ox, oy, w, h, stride, 1, y0, y1, y0, temp2_base_y);
}

void AccumCh2CoreRange(float* weight, float* wdst0, float* wdst1, float* maxw, const float* s0_bwd,
                       const float* s1_bwd, const float* s0_fwd, const float* s1_fwd, const float* temp1_core,
                       const float* temp2, int ox, int oy, int w, int h, int stride, int y0, int y1,
                       int temp2_base_y) {
    accum_inner(weight, wdst0, wdst1, nullptr, maxw, s0_bwd, s1_bwd, nullptr, s0_fwd, s1_fwd, nullptr, temp1_core,
                temp2, ox, oy, w, h, stride, 2, y0, y1, y0, temp2_base_y);
}

void AccumCh3CoreRange(float* weight, float* wdst0, float* wdst1, float* wdst2, float* maxw, const float* s0_bwd,
                       const float* s1_bwd, const float* s2_bwd, const float* s0_fwd, const float* s1_fwd,
                       const float* s2_fwd, const float* temp1_core, const float* temp2, int ox, int oy, int w, int h,
                       int stride, int y0, int y1, int temp2_base_y) {
    accum_inner(weight, wdst0, wdst1, wdst2, maxw, s0_bwd, s1_bwd, s2_bwd, s0_fwd, s1_fwd, s2_fwd, temp1_core, temp2,
                ox, oy, w, h, stride, 3, y0, y1, y0, temp2_base_y);
}

void FinishCh1(float* dst, const float* src, const float* weight, const float* wdst, const float* maxw, float wref,
               int w, int h, int stride) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const auto vwref = hn::Set(d, wref);
    for (int y = 0; y < h; ++y) {
        int x = 0;
        float* o = dst + y * stride;
        const float* s = src + y * stride;
        const float* wt = weight + y * stride;
        const float* wd = wdst + y * stride;
        const float* mw = maxw + y * stride;
        for (; x + N <= w; x += N) {
            const auto mul = hn::Mul(vwref, hn::LoadU(d, mw + x));
            const auto den = hn::Add(mul, hn::LoadU(d, wt + x));
            hn::StoreU(hn::Div(hn::MulAdd(mul, hn::LoadU(d, s + x), hn::LoadU(d, wd + x)), den), d, o + x);
        }
        for (; x < w; ++x) {
            const float mul = wref * mw[x];
            o[x] = (mul * s[x] + wd[x]) / (mul + wt[x]);
        }
    }
}

void FinishCh2(float* d0, float* d1, const float* s0, const float* s1, const float* weight, const float* wdst0,
               const float* wdst1, const float* maxw, float wref, int w, int h, int stride) {
    FinishCh1(d0, s0, weight, wdst0, maxw, wref, w, h, stride);
    FinishCh1(d1, s1, weight, wdst1, maxw, wref, w, h, stride);
}

void FinishCh3(float* d0, float* d1, float* d2, const float* s0, const float* s1, const float* s2,
               const float* weight, const float* wdst0, const float* wdst1, const float* wdst2, const float* maxw,
               float wref, int w, int h, int stride) {
    FinishCh1(d0, s0, weight, wdst0, maxw, wref, w, h, stride);
    FinishCh1(d1, s1, weight, wdst1, maxw, wref, w, h, stride);
    FinishCh1(d2, s2, weight, wdst2, maxw, wref, w, h, stride);
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {

namespace {

inline int clampi_mixed(int x, int lo, int hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// Mixed-stride frames are uncommon on the hot path (the normal VapourSynth
// allocator gives all selected planes the same stride), so keep this fallback
// scalar and leave the equal-stride calls on the Highway kernels above.
void distance_strided_impl(float* dst, const float* const* center, const int* center_strides,
                           const float* const* neighbor, const int* neighbor_strides, ChannelMode mode, int ox, int oy,
                           int w, int h, int dst_stride) {
    if (!dst || !center || !center_strides || !neighbor || !neighbor_strides || w < 1 || h < 1 || dst_stride < w) {
        return;
    }
    const int first = mode == ChannelMode::UV ? 1 : 0;
    const int nch = mode == ChannelMode::Y ? 1 : (mode == ChannelMode::UV ? 2 : 3);
    for (int c = 0; c < nch; ++c) {
        const int p = first + c;
        if (!center[p] || !neighbor[p] || center_strides[p] < w || neighbor_strides[p] < w) {
            return;
        }
    }
    for (int y = 0; y < h; ++y) {
        const int ny = clampi_mixed(y + oy, 0, h - 1);
        for (int x = 0; x < w; ++x) {
            const int nx = clampi_mixed(x + ox, 0, w - 1);
            float value = 0.f;
            if (mode == ChannelMode::Y) {
                const float d = center[0][y * center_strides[0] + x] -
                                neighbor[0][ny * neighbor_strides[0] + nx];
                value = 3.f * d * d;
            } else if (mode == ChannelMode::UV) {
                const float d1 = center[1][y * center_strides[1] + x] -
                                 neighbor[1][ny * neighbor_strides[1] + nx];
                const float d2 = center[2][y * center_strides[2] + x] -
                                 neighbor[2][ny * neighbor_strides[2] + nx];
                value = 1.5f * (d1 * d1 + d2 * d2);
            } else if (mode == ChannelMode::YUV) {
                for (int p = 0; p < 3; ++p) {
                    const float d = center[p][y * center_strides[p] + x] -
                                    neighbor[p][ny * neighbor_strides[p] + nx];
                    value += d * d;
                }
            } else if (mode == ChannelMode::RGB) {
                const float u1 = center[0][y * center_strides[0] + x];
                const float v1 = neighbor[0][ny * neighbor_strides[0] + nx];
                const float u2 = center[1][y * center_strides[1] + x];
                const float v2 = neighbor[1][ny * neighbor_strides[1] + nx];
                const float u3 = center[2][y * center_strides[2] + x];
                const float v3 = neighbor[2][ny * neighbor_strides[2] + nx];
                const float m_red = (u1 + v1) / 6.f;
                const float d1 = u1 - v1;
                const float d2 = u2 - v2;
                const float d3 = u3 - v3;
                value = (2.f / 3.f + m_red) * d1 * d1 + (4.f / 3.f) * d2 * d2 + (1.f - m_red) * d3 * d3;
            } else {
                return;
            }
            dst[y * dst_stride + x] = value;
        }
    }
}

void accum_strided_impl(float* weight, float* wdst0, float* wdst1, float* wdst2, float* maxw,
                        const float* const* src_bwd, const int* src_bwd_strides,
                        const float* const* src_fwd, const int* src_fwd_strides,
                        const float* temp1, const float* temp2, int nch, int ox, int oy, int w, int h, int stride,
                        int y0, int y1, int temp1_base_y) {
    if (!weight || !maxw || !src_bwd || !src_bwd_strides || !src_fwd || !src_fwd_strides || !temp1 || !temp2 ||
        w < 1 || h < 1 || stride < w || y0 < 0 || y1 < y0 || y1 > h || temp1_base_y < 0) {
        return;
    }
    nch = std::max(1, std::min(3, nch));
    float* wd[3] = {wdst0, wdst1, wdst2};
    for (int c = 0; c < nch; ++c) {
        if (!wd[c] || !src_bwd[c] || !src_fwd[c] || src_bwd_strides[c] < w || src_fwd_strides[c] < w) {
            return;
        }
    }
    for (int y = y0; y < y1; ++y) {
        const int out_y = y - y0;
        const int temp_y = y - temp1_base_y;
        if (temp_y < 0) {
            return;
        }
        for (int x = 0; x < w; ++x) {
            const int out_idx = out_y * stride + x;
            const int temp_idx = temp_y * stride + x;
            const int mq_y = clampi_mixed(y - oy, 0, h - 1);
            const int pq_y = clampi_mixed(y + oy, 0, h - 1);
            const int mq_x = clampi_mixed(x - ox, 0, w - 1);
            const int pq_x = clampi_mixed(x + ox, 0, w - 1);
            const float u = temp1[temp_idx];
            const float um = temp2[mq_y * stride + mq_x];
            weight[out_idx] += u + um;
            maxw[out_idx] = std::max(maxw[out_idx], std::max(u, um));
            for (int c = 0; c < nch; ++c) {
                const float b = src_bwd[c][pq_y * src_bwd_strides[c] + pq_x];
                const float f = src_fwd[c][mq_y * src_fwd_strides[c] + mq_x];
                wd[c][out_idx] += u * b + um * f;
            }
        }
    }
}

void finish_strided_impl(float* const* dst, const int* dst_strides, const float* const* src, const int* src_strides,
                         const float* weight, const float* const* wdst, const float* maxw, float wref, int nch, int w,
                         int h, int map_stride) {
    if (!dst || !dst_strides || !src || !src_strides || !weight || !wdst || !maxw || w < 1 || h < 1 ||
        map_stride < w) {
        return;
    }
    nch = std::max(1, std::min(3, nch));
    for (int c = 0; c < nch; ++c) {
        if (!dst[c] || !src[c] || !wdst[c] || dst_strides[c] < w || src_strides[c] < w) {
            return;
        }
    }
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int map_idx = y * map_stride + x;
            const float mul = wref * maxw[map_idx];
            const float denom = mul + weight[map_idx];
            for (int c = 0; c < nch; ++c) {
                dst[c][y * dst_strides[c] + x] =
                    (mul * src[c][y * src_strides[c] + x] + wdst[c][map_idx]) / denom;
            }
        }
    }
}

}  // namespace

HWY_EXPORT(DistanceLuma);
HWY_EXPORT(DistanceChroma);
HWY_EXPORT(DistanceYUV);
HWY_EXPORT(DistanceRGB);
HWY_EXPORT(Horizontal);
HWY_EXPORT(VerticalWelsch);
HWY_EXPORT(VerticalWelschRange);
HWY_EXPORT(AccumCh1);
HWY_EXPORT(AccumCh2);
HWY_EXPORT(AccumCh3);
HWY_EXPORT(AccumCh1Range);
HWY_EXPORT(AccumCh2Range);
HWY_EXPORT(AccumCh3Range);
HWY_EXPORT(AccumCh1CoreRange);
HWY_EXPORT(AccumCh2CoreRange);
HWY_EXPORT(AccumCh3CoreRange);
HWY_EXPORT(FinishCh1);
HWY_EXPORT(FinishCh2);
HWY_EXPORT(FinishCh3);

void nlm_distance_luma_f32(float* dst, const float* center, const float* neighbor, int ox, int oy, int w, int h,
                           int stride) {
    HWY_DYNAMIC_DISPATCH(DistanceLuma)(dst, center, neighbor, ox, oy, w, h, stride);
}
void nlm_distance_chroma_f32(float* dst, const float* c1, const float* c2, const float* n1, const float* n2, int ox,
                             int oy, int w, int h, int stride) {
    HWY_DYNAMIC_DISPATCH(DistanceChroma)(dst, c1, c2, n1, n2, ox, oy, w, h, stride);
}
void nlm_distance_yuv_f32(float* dst, const float* c0, const float* c1, const float* c2, const float* n0,
                          const float* n1, const float* n2, int ox, int oy, int w, int h, int stride) {
    HWY_DYNAMIC_DISPATCH(DistanceYUV)(dst, c0, c1, c2, n0, n1, n2, ox, oy, w, h, stride);
}
void nlm_distance_rgb_f32(float* dst, const float* c0, const float* c1, const float* c2, const float* n0,
                          const float* n1, const float* n2, int ox, int oy, int w, int h, int stride) {
    HWY_DYNAMIC_DISPATCH(DistanceRGB)(dst, c0, c1, c2, n0, n1, n2, ox, oy, w, h, stride);
}
void nlm_distance_strided_f32(float* dst, const float* const* center, const int* center_strides,
                              const float* const* neighbor, const int* neighbor_strides, ChannelMode mode, int ox,
                              int oy, int w, int h, int dst_stride) {
    distance_strided_impl(dst, center, center_strides, neighbor, neighbor_strides, mode, ox, oy, w, h, dst_stride);
}
void nlm_horizontal(float* dst, const float* src, int s, int w, int h, int stride) {
    HWY_DYNAMIC_DISPATCH(Horizontal)(dst, src, s, w, h, stride);
}
void nlm_vertical_welsch(float* dst, const float* src, int radius, float h2_inv_norm, int w, int h, int stride,
                         float* buffer) {
    HWY_DYNAMIC_DISPATCH(VerticalWelsch)(dst, src, radius, h2_inv_norm, w, h, stride, buffer);
}
void nlm_vertical_welsch_range(float* dst, const float* src, int radius, float h2_inv_norm, int w, int h, int stride,
                               int y0, int y1, float* buffer) {
    HWY_DYNAMIC_DISPATCH(VerticalWelschRange)(dst, src, radius, h2_inv_norm, w, h, stride, y0, y1, buffer);
}
void nlm_accum_ch1(float* weight, float* wdst, float* maxw, const float* src_bwd, const float* src_fwd,
                   const float* temp1, const float* temp2, int ox, int oy, int w, int h, int stride) {
    HWY_DYNAMIC_DISPATCH(AccumCh1)(weight, wdst, maxw, src_bwd, src_fwd, temp1, temp2, ox, oy, w, h, stride);
}
void nlm_accum_ch2(float* weight, float* wdst0, float* wdst1, float* maxw, const float* s0_bwd, const float* s1_bwd,
                   const float* s0_fwd, const float* s1_fwd, const float* temp1, const float* temp2, int ox, int oy,
                   int w, int h, int stride) {
    HWY_DYNAMIC_DISPATCH(AccumCh2)(weight, wdst0, wdst1, maxw, s0_bwd, s1_bwd, s0_fwd, s1_fwd, temp1, temp2, ox, oy, w,
                                   h, stride);
}
void nlm_accum_ch3(float* weight, float* wdst0, float* wdst1, float* wdst2, float* maxw, const float* s0_bwd,
                   const float* s1_bwd, const float* s2_bwd, const float* s0_fwd, const float* s1_fwd,
                   const float* s2_fwd, const float* temp1, const float* temp2, int ox, int oy, int w, int h,
                   int stride) {
    HWY_DYNAMIC_DISPATCH(AccumCh3)(weight, wdst0, wdst1, wdst2, maxw, s0_bwd, s1_bwd, s2_bwd, s0_fwd, s1_fwd, s2_fwd,
                                   temp1, temp2, ox, oy, w, h, stride);
}
void nlm_accum_ch1_range(float* weight, float* wdst, float* maxw, const float* src_bwd, const float* src_fwd,
                         const float* temp1, const float* temp2, int ox, int oy, int w, int h, int stride, int y0,
                         int y1) {
    HWY_DYNAMIC_DISPATCH(AccumCh1Range)(weight, wdst, maxw, src_bwd, src_fwd, temp1, temp2, ox, oy, w, h, stride, y0,
                                        y1);
}
void nlm_accum_ch2_range(float* weight, float* wdst0, float* wdst1, float* maxw, const float* s0_bwd,
                         const float* s1_bwd, const float* s0_fwd, const float* s1_fwd, const float* temp1,
                         const float* temp2, int ox, int oy, int w, int h, int stride, int y0, int y1) {
    HWY_DYNAMIC_DISPATCH(AccumCh2Range)(weight, wdst0, wdst1, maxw, s0_bwd, s1_bwd, s0_fwd, s1_fwd, temp1, temp2, ox,
                                        oy, w, h, stride, y0, y1);
}
void nlm_accum_ch3_range(float* weight, float* wdst0, float* wdst1, float* wdst2, float* maxw, const float* s0_bwd,
                         const float* s1_bwd, const float* s2_bwd, const float* s0_fwd, const float* s1_fwd,
                         const float* s2_fwd, const float* temp1, const float* temp2, int ox, int oy, int w, int h,
                         int stride, int y0, int y1) {
    HWY_DYNAMIC_DISPATCH(AccumCh3Range)(weight, wdst0, wdst1, wdst2, maxw, s0_bwd, s1_bwd, s2_bwd, s0_fwd, s1_fwd,
                                        s2_fwd, temp1, temp2, ox, oy, w, h, stride, y0, y1);
}
void nlm_accum_ch1_core_range(float* weight, float* wdst, float* maxw, const float* src_bwd, const float* src_fwd,
                              const float* temp1_core, const float* temp2, int ox, int oy, int w, int h, int stride,
                              int y0, int y1, int temp2_base_y) {
    HWY_DYNAMIC_DISPATCH(AccumCh1CoreRange)(weight, wdst, maxw, src_bwd, src_fwd, temp1_core, temp2, ox, oy, w, h,
                                            stride, y0, y1, temp2_base_y);
}
void nlm_accum_ch2_core_range(float* weight, float* wdst0, float* wdst1, float* maxw, const float* s0_bwd,
                              const float* s1_bwd, const float* s0_fwd, const float* s1_fwd, const float* temp1_core,
                              const float* temp2, int ox, int oy, int w, int h, int stride, int y0, int y1,
                              int temp2_base_y) {
    HWY_DYNAMIC_DISPATCH(AccumCh2CoreRange)(weight, wdst0, wdst1, maxw, s0_bwd, s1_bwd, s0_fwd, s1_fwd, temp1_core,
                                            temp2, ox, oy, w, h, stride, y0, y1, temp2_base_y);
}
void nlm_accum_ch3_core_range(float* weight, float* wdst0, float* wdst1, float* wdst2, float* maxw,
                              const float* s0_bwd, const float* s1_bwd, const float* s2_bwd, const float* s0_fwd,
                              const float* s1_fwd, const float* s2_fwd, const float* temp1_core, const float* temp2,
                              int ox, int oy, int w, int h, int stride, int y0, int y1, int temp2_base_y) {
    HWY_DYNAMIC_DISPATCH(AccumCh3CoreRange)(weight, wdst0, wdst1, wdst2, maxw, s0_bwd, s1_bwd, s2_bwd, s0_fwd,
                                            s1_fwd, s2_fwd, temp1_core, temp2, ox, oy, w, h, stride, y0, y1,
                                            temp2_base_y);
}
void nlm_accum_strided(float* weight, float* wdst0, float* wdst1, float* wdst2, float* maxw,
                       const float* const* src_bwd, const int* src_bwd_strides,
                       const float* const* src_fwd, const int* src_fwd_strides,
                       const float* temp1, const float* temp2, int nch, int ox, int oy, int w, int h, int stride,
                       int y0, int y1, int temp1_base_y) {
    accum_strided_impl(weight, wdst0, wdst1, wdst2, maxw, src_bwd, src_bwd_strides, src_fwd, src_fwd_strides, temp1,
                       temp2, nch, ox, oy, w, h, stride, y0, y1, temp1_base_y);
}
void nlm_finish_ch1(float* dst, const float* src, const float* weight, const float* wdst, const float* maxw,
                    float wref, int w, int h, int stride) {
    HWY_DYNAMIC_DISPATCH(FinishCh1)(dst, src, weight, wdst, maxw, wref, w, h, stride);
}
void nlm_finish_ch2(float* d0, float* d1, const float* s0, const float* s1, const float* weight, const float* wdst0,
                    const float* wdst1, const float* maxw, float wref, int w, int h, int stride) {
    HWY_DYNAMIC_DISPATCH(FinishCh2)(d0, d1, s0, s1, weight, wdst0, wdst1, maxw, wref, w, h, stride);
}
void nlm_finish_ch3(float* d0, float* d1, float* d2, const float* s0, const float* s1, const float* s2,
                    const float* weight, const float* wdst0, const float* wdst1, const float* wdst2, const float* maxw,
                    float wref, int w, int h, int stride) {
    HWY_DYNAMIC_DISPATCH(FinishCh3)(d0, d1, d2, s0, s1, s2, weight, wdst0, wdst1, wdst2, maxw, wref, w, h, stride);
}
void nlm_finish_strided(float* const* dst, const int* dst_strides, const float* const* src, const int* src_strides,
                        const float* weight, const float* const* wdst, const float* maxw, float wref, int nch, int w,
                        int h, int map_stride) {
    finish_strided_impl(dst, dst_strides, src, src_strides, weight, wdst, maxw, wref, nch, w, h, map_stride);
}
}  // namespace nss
#endif
