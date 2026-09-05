#include "nss/cpu_api.hpp"
#include "cpu/hwy_config.hpp"

#include <algorithm>
#include <cstring>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/bm/pack.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

static int clampi(int x, int lo, int hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

void PackPatch(float* col, int lda, const float* src, int stride, int x, int y, int block, int width,
               int height) {
    (void)lda;
    const bool inside = x >= 0 && y >= 0 && x + block <= width && y + block <= height;
    if (inside) {
        for (int py = 0; py < block; ++py) {
            std::memcpy(col + py * block, src + (y + py) * stride + x, static_cast<size_t>(block) * sizeof(float));
        }
        return;
    }
    for (int py = 0; py < block; ++py) {
        const int yy = clampi(y + py, 0, height - 1);
        for (int px = 0; px < block; ++px) {
            const int xx = clampi(x + px, 0, width - 1);
            col[py * block + px] = src[yy * stride + xx];
        }
    }
}

void UnpackPatch(float* num, float* den, int stride, int x, int y, const float* col, int block, int width,
                 int height, float w) {
    const int x0 = std::max(x, 0);
    const int y0 = std::max(y, 0);
    const int x1 = std::min(x + block, width);
    const int y1 = std::min(y + block, height);
    if (x0 >= x1 || y0 >= y1) {
        return;
    }
#if HWY_MAX_BYTES >= 32
    const hn::FixedTag<float, 8> d8;
    const auto vw8 = hn::Set(d8, w);
#endif
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const auto vw = hn::Set(d, w);
    for (int yy = y0; yy < y1; ++yy) {
        const int py = yy - y;
        const float* row = col + py * block + (x0 - x);
        float* n = num + yy * stride + x0;
        float* de = den + yy * stride + x0;
        const int len = x1 - x0;
        int px = 0;
#if HWY_MAX_BYTES >= 32
        if (len == 8) {
            hn::StoreU(hn::MulAdd(vw8, hn::LoadU(d8, row), hn::LoadU(d8, n)), d8, n);
            hn::StoreU(hn::Add(hn::LoadU(d8, de), vw8), d8, de);
            continue;
        }
#endif
        for (; px + N <= len; px += N) {
            hn::StoreU(hn::MulAdd(vw, hn::LoadU(d, row + px), hn::LoadU(d, n + px)), d, n + px);
            hn::StoreU(hn::Add(hn::LoadU(d, de + px), vw), d, de + px);
        }
#if HWY_MAX_BYTES >= 32
        for (; px + 8 <= len; px += 8) {
            hn::StoreU(hn::MulAdd(vw8, hn::LoadU(d8, row + px), hn::LoadU(d8, n + px)), d8, n + px);
            hn::StoreU(hn::Add(hn::LoadU(d8, de + px), vw8), d8, de + px);
        }
#endif
        for (; px < len; ++px) {
            n[px] += w * row[px];
            de[px] += w;
        }
    }
}

// BM3D's fixed spatial paths only produce in-bounds matches. Keep the generic
// clamped implementation as the fallback, but make the common 4/12 rows a
// single fixed-width SIMD operation so b4 and b12 do not fall through scalar
// tails on AVX-512.
void UnpackPatchFixed(float* num, float* den, int stride, int x, int y, const float* col, int block, int width,
                      int height, float w) {
    const bool inside = x >= 0 && y >= 0 && x + block <= width && y + block <= height;
    if (!inside) {
        UnpackPatch(num, den, stride, x, y, col, block, width, height, w);
        return;
    }
#if HWY_MAX_BYTES >= 16
    if (block == 4) {
        const hn::FixedTag<float, 4> d4;
        const auto vw = hn::Set(d4, w);
        for (int row = 0; row < 4; ++row) {
            float* n = num + (y + row) * stride + x;
            float* de = den + (y + row) * stride + x;
            const auto values = hn::LoadU(d4, col + row * 4);
            hn::StoreU(hn::MulAdd(vw, values, hn::LoadU(d4, n)), d4, n);
            hn::StoreU(hn::Add(hn::LoadU(d4, de), vw), d4, de);
        }
        return;
    }
#endif
#if HWY_MAX_BYTES >= 64
    if (block == 12) {
        const hn::FixedTag<float, 16> d16;
        const auto vw = hn::Set(d16, w);
        const std::size_t count = static_cast<std::size_t>(block);
        for (int row = 0; row < block; ++row) {
            float* n = num + (y + row) * stride + x;
            float* de = den + (y + row) * stride + x;
            const auto values = hn::LoadN(d16, col + row * block, count);
            hn::StoreN(hn::MulAdd(vw, values, hn::LoadN(d16, n, count)), d16, n, count);
            hn::StoreN(hn::Add(hn::LoadN(d16, de, count), vw), d16, de, count);
        }
        return;
    }
#endif
    UnpackPatch(num, den, stride, x, y, col, block, width, height, w);
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(PackPatch);
HWY_EXPORT(UnpackPatch);
HWY_EXPORT(UnpackPatchFixed);

void pack_patch(float* col, int lda, const float* src, int stride, int x, int y,
                int block, int width, int height) {
    HWY_DYNAMIC_DISPATCH(PackPatch)(col, lda, src, stride, x, y, block, width, height);
}

void unpack_patch(float* num, float* den, int stride, int x, int y,
                  const float* col, int block, int width, int height, float w) {
    HWY_DYNAMIC_DISPATCH(UnpackPatch)(num, den, stride, x, y, col, block, width, height, w);
}

void unpack_patch_fixed(float* num, float* den, int stride, int x, int y,
                        const float* col, int block, int width, int height, float w) {
    HWY_DYNAMIC_DISPATCH(UnpackPatchFixed)(num, den, stride, x, y, col, block, width, height, w);
}

}  // namespace nss
#endif
