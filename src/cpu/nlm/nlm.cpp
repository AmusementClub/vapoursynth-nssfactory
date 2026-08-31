#include "nss/cpu_api.hpp"
#include "cpu/hwy_config.hpp"

#include <algorithm>
#include <cmath>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/nlm/nlm.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
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

static inline void sub_row(float* dst, const float* src, int n) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    int x = 0;
    for (; x + N <= n; x += N) {
        hn::StoreU(hn::Sub(hn::LoadU(d, dst + x), hn::LoadU(d, src + x)), d, dst + x);
    }
    for (; x < n; ++x) {
        dst[x] -= src[x];
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

static inline void exp_scale_row(float* dst, const float* buf, float scale, int n) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const auto vs = hn::Set(d, scale);
    int x = 0;
    for (; x + N <= n; x += N) {
        hn::StoreU(hn::Exp(d, hn::Mul(hn::LoadU(d, buf + x), vs)), d, dst + x);
    }
    for (; x < n; ++x) {
        dst[x] = std::exp(buf[x] * scale);
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
            auto acc = hn::LoadU(d, row + x - S);
            for (int j = -S + 1; j <= S; ++j) {
                acc = hn::Add(acc, hn::LoadU(d, row + x + j));
            }
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

void VerticalWelsch(float* dst, const float* src, int radius, float h2_inv_norm, int w, int h, int stride,
                    float* buffer) {
    const float nscale = -h2_inv_norm;
    scale_row(buffer, src, static_cast<float>(radius), w);
    for (int y = 0; y < radius; ++y) {
        const int yy = std::min(y, h - 1);
        add_row(buffer, src + yy * stride, w);
    }
    for (int y = 0; y < std::min(radius, h); ++y) {
        const int yadd = std::min(y + radius, h - 1);
        add_row(buffer, src + yadd * stride, w);
        exp_scale_row(dst + y * stride, buffer, nscale, w);
        sub_row(buffer, src, w);
    }
    if (h > radius) {
        for (int y = radius; y < h - radius; ++y) {
            add_row(buffer, src + (y + radius) * stride, w);
            exp_scale_row(dst + y * stride, buffer, nscale, w);
            sub_row(buffer, src + (y - radius) * stride, w);
        }
        for (int y = std::max(h - radius, radius); y < h; ++y) {
            const int yadd = std::min(y + radius, h - 1);
            add_row(buffer, src + yadd * stride, w);
            exp_scale_row(dst + y * stride, buffer, nscale, w);
            sub_row(buffer, src + (y - radius) * stride, w);
        }
    }
}

static void accum_inner(float* weight, float* wdst0, float* wdst1, float* wdst2, float* maxw,
                        const float* s0_bwd, const float* s1_bwd, const float* s2_bwd, const float* s0_fwd,
                        const float* s1_fwd, const float* s2_fwd, const float* temp1, const float* temp2,
                        int ox, int oy, int w, int h, int stride, int nch) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const int start_x = std::abs(ox);
    const int end_x = w - std::abs(ox);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < start_x; ++x) {
            const int idx = y * stride + x;
            const float u4 = temp1[idx];
            const int mq = clampi(y - oy, 0, h - 1) * stride + clampi(x - ox, 0, w - 1);
            const float u4_mq = temp2[mq];
            weight[idx] += u4 + u4_mq;
            maxw[idx] = std::max(std::max(u4, u4_mq), maxw[idx]);
            const int pq = clampi(y + oy, 0, h - 1) * stride + clampi(x + ox, 0, w - 1);
            wdst0[idx] += u4 * s0_bwd[pq] + u4_mq * s0_fwd[mq];
            if (nch > 1) {
                wdst1[idx] += u4 * s1_bwd[pq] + u4_mq * s1_fwd[mq];
            }
            if (nch > 2) {
                wdst2[idx] += u4 * s2_bwd[pq] + u4_mq * s2_fwd[mq];
            }
        }
        if (end_x > start_x) {
            const int ypq = clampi(y + oy, 0, h - 1);
            const int ymq = clampi(y - oy, 0, h - 1);
            const int off = y * stride + start_x;
            const int mq0 = ymq * stride + start_x - ox;
            const int pq0 = ypq * stride + start_x + ox;
            const int n = end_x - start_x;
            int x = 0;
            for (; x + N <= n; x += N) {
                const auto u4 = hn::LoadU(d, temp1 + off + x);
                const auto u4m = hn::LoadU(d, temp2 + mq0 + x);
                hn::StoreU(hn::Add(hn::LoadU(d, weight + off + x), hn::Add(u4, u4m)), d, weight + off + x);
                hn::StoreU(hn::Max(hn::Max(u4, u4m), hn::LoadU(d, maxw + off + x)), d, maxw + off + x);
                auto acc0 = hn::MulAdd(u4, hn::LoadU(d, s0_bwd + pq0 + x), hn::Mul(u4m, hn::LoadU(d, s0_fwd + mq0 + x)));
                hn::StoreU(hn::Add(hn::LoadU(d, wdst0 + off + x), acc0), d, wdst0 + off + x);
                if (nch > 1) {
                    auto acc1 =
                        hn::MulAdd(u4, hn::LoadU(d, s1_bwd + pq0 + x), hn::Mul(u4m, hn::LoadU(d, s1_fwd + mq0 + x)));
                    hn::StoreU(hn::Add(hn::LoadU(d, wdst1 + off + x), acc1), d, wdst1 + off + x);
                }
                if (nch > 2) {
                    auto acc2 =
                        hn::MulAdd(u4, hn::LoadU(d, s2_bwd + pq0 + x), hn::Mul(u4m, hn::LoadU(d, s2_fwd + mq0 + x)));
                    hn::StoreU(hn::Add(hn::LoadU(d, wdst2 + off + x), acc2), d, wdst2 + off + x);
                }
            }
            for (; x < n; ++x) {
                const float u4 = temp1[off + x];
                const float u4_mq = temp2[mq0 + x];
                weight[off + x] += u4 + u4_mq;
                maxw[off + x] = std::max(std::max(u4, u4_mq), maxw[off + x]);
                wdst0[off + x] += u4 * s0_bwd[pq0 + x] + u4_mq * s0_fwd[mq0 + x];
                if (nch > 1) {
                    wdst1[off + x] += u4 * s1_bwd[pq0 + x] + u4_mq * s1_fwd[mq0 + x];
                }
                if (nch > 2) {
                    wdst2[off + x] += u4 * s2_bwd[pq0 + x] + u4_mq * s2_fwd[mq0 + x];
                }
            }
        }
        for (int x = std::max(end_x, start_x); x < w; ++x) {
            const int idx = y * stride + x;
            const float u4 = temp1[idx];
            const int mq = clampi(y - oy, 0, h - 1) * stride + clampi(x - ox, 0, w - 1);
            const float u4_mq = temp2[mq];
            weight[idx] += u4 + u4_mq;
            maxw[idx] = std::max(std::max(u4, u4_mq), maxw[idx]);
            const int pq = clampi(y + oy, 0, h - 1) * stride + clampi(x + ox, 0, w - 1);
            wdst0[idx] += u4 * s0_bwd[pq] + u4_mq * s0_fwd[mq];
            if (nch > 1) {
                wdst1[idx] += u4 * s1_bwd[pq] + u4_mq * s1_fwd[mq];
            }
            if (nch > 2) {
                wdst2[idx] += u4 * s2_bwd[pq] + u4_mq * s2_fwd[mq];
            }
        }
    }
}

void AccumCh1(float* weight, float* wdst, float* maxw, const float* src_bwd, const float* src_fwd,
              const float* temp1, const float* temp2, int ox, int oy, int w, int h, int stride) {
    accum_inner(weight, wdst, nullptr, nullptr, maxw, src_bwd, nullptr, nullptr, src_fwd, nullptr, nullptr, temp1,
                temp2, ox, oy, w, h, stride, 1);
}

void AccumCh2(float* weight, float* wdst0, float* wdst1, float* maxw, const float* s0_bwd, const float* s1_bwd,
              const float* s0_fwd, const float* s1_fwd, const float* temp1, const float* temp2, int ox, int oy, int w,
              int h, int stride) {
    accum_inner(weight, wdst0, wdst1, nullptr, maxw, s0_bwd, s1_bwd, nullptr, s0_fwd, s1_fwd, nullptr, temp1, temp2, ox,
                oy, w, h, stride, 2);
}

void AccumCh3(float* weight, float* wdst0, float* wdst1, float* wdst2, float* maxw, const float* s0_bwd,
              const float* s1_bwd, const float* s2_bwd, const float* s0_fwd, const float* s1_fwd, const float* s2_fwd,
              const float* temp1, const float* temp2, int ox, int oy, int w, int h, int stride) {
    accum_inner(weight, wdst0, wdst1, wdst2, maxw, s0_bwd, s1_bwd, s2_bwd, s0_fwd, s1_fwd, s2_fwd, temp1, temp2, ox, oy,
                w, h, stride, 3);
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
HWY_EXPORT(DistanceLuma);
HWY_EXPORT(DistanceChroma);
HWY_EXPORT(DistanceYUV);
HWY_EXPORT(DistanceRGB);
HWY_EXPORT(Horizontal);
HWY_EXPORT(VerticalWelsch);
HWY_EXPORT(AccumCh1);
HWY_EXPORT(AccumCh2);
HWY_EXPORT(AccumCh3);
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
void nlm_horizontal(float* dst, const float* src, int s, int w, int h, int stride) {
    HWY_DYNAMIC_DISPATCH(Horizontal)(dst, src, s, w, h, stride);
}
void nlm_vertical_welsch(float* dst, const float* src, int radius, float h2_inv_norm, int w, int h, int stride,
                         float* buffer) {
    HWY_DYNAMIC_DISPATCH(VerticalWelsch)(dst, src, radius, h2_inv_norm, w, h, stride, buffer);
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
}  // namespace nss
#endif
