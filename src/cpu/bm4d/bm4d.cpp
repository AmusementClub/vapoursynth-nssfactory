// SPDX-License-Identifier: GPL-2.0-only
// BM4D model version 1 (include/nss/params/bm4d.hpp), written from the paper.
// Pinned TU (no contraction): every fused operation below is explicit, and
// every sum has the order the model states, so the output does not depend on
// the vector width.
#include "nss/cpu_bm4d.hpp"
#include "nss/params/bm4d.hpp"
#include "cpu/hwy_config.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/bm4d/bm4d.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

#ifndef NSS_BM4D_ONCE
#define NSS_BM4D_ONCE
namespace nss {
namespace bm4d_detail {

constexpr int kMaxCube = 5;
constexpr int kMaxCube3 = kMaxCube * kMaxCube * kMaxCube;
constexpr int kMaxGroup = 32;

inline std::size_t align16(std::size_t n) { return (n + 15) & ~std::size_t{15}; }

// Reference positions along an edge of n: 0, step, ... and n - L.
inline int position_count(int n, int L, int step) {
    if (n < L) return 0;
    const int regular = (n - L) / step + 1;
    return regular + ((n - L) % step != 0 ? 1 : 0);
}
inline int position(int n, int L, int step, int i) {
    const int p = i * step;
    return p > n - L ? n - L : p;
}

struct Transforms {
    float dct[2][kMaxCube][kMaxCube];  // [L == 5][k][i], orthonormal DCT-II
    float kaiser[2][kMaxCube];
    Transforms() {
        const double pi = 3.14159265358979323846;
        for (int s = 0; s < 2; ++s) {
            const int L = s ? 5 : 4;
            for (int k = 0; k < L; ++k)
                for (int i = 0; i < L; ++i) {
                    double v = std::cos(pi * (2 * i + 1) * k / (2.0 * L)) * std::sqrt(2.0 / L);
                    if (k == 0) v /= std::sqrt(2.0);
                    dct[s][k][i] = static_cast<float>(v);
                }
            // np.kaiser: I0(beta sqrt(1 - (2n / (L - 1) - 1)^2)) / I0(beta)
            auto i0 = [](double x) {
                double sum = 1, term = 1;
                for (int k = 1; k < 64; ++k) {
                    term *= (x / (2 * k)) * (x / (2 * k));
                    sum += term;
                }
                return sum;
            };
            for (int n = 0; n < L; ++n) {
                const double r = 2.0 * n / (L - 1) - 1.0;
                kaiser[s][n] = static_cast<float>(i0(kBm4dKaiserBeta * std::sqrt(1 - r * r)) / i0(kBm4dKaiserBeta));
            }
        }
    }
};

inline const Transforms& transforms() {
    static const Transforms t;
    return t;
}

// The best matches of every reference cube: kept sorted by distance.
struct Matches {
    float* dist;
    std::int32_t* pos;  // cube origin (z h + y) w + x
    std::int32_t* count;
    int group;
    void insert(int r, float d, std::int32_t p) {
        float* dr = dist + static_cast<std::size_t>(r) * group;
        std::int32_t* pr = pos + static_cast<std::size_t>(r) * group;
        int c = count[r];
        if (c == group) {
            if (!(d < dr[group - 1])) return;
            --c;
        }
        int i = c;
        while (i > 0 && d < dr[i - 1]) {
            dr[i] = dr[i - 1];
            pr[i] = pr[i - 1];
            --i;
        }
        dr[i] = d;
        pr[i] = p;
        count[r] = c + 1;
    }
};

}  // namespace bm4d_detail
}  // namespace nss
#endif  // NSS_BM4D_ONCE

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;
using namespace bm4d_detail;

struct Volume {
    int t, h, w;
    std::size_t at(int z, int y, int x) const { return (static_cast<std::size_t>(z) * h + y) * w + x; }
};

struct Refs {
    int nz, ny, nx;
    int L, step;
    int count() const { return nz * ny * nx; }
};

// A[y][x] = sum over k < L (in order) of (V[z0 + k][y][x] - V[z0 + k + dz][y + dy][x + dx])^2
// for the rows and columns where both exist.
static HWY_NOINLINE void Slab(const float* V, const Volume& vol, int L, int z0, int dz, int dy, int dx, int y_lo,
                              int y_hi, int x_lo, int x_hi, float* A) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    for (int y = y_lo; y < y_hi; ++y) {
        float* a = A + static_cast<std::size_t>(y) * vol.w;
        int x = x_lo;
        for (; x + N <= x_hi; x += N) {
            auto acc = hn::Zero(d);
            for (int k = 0; k < L; ++k) {
                const auto p = hn::LoadU(d, V + vol.at(z0 + k, y, x));
                const auto q = hn::LoadU(d, V + vol.at(z0 + k + dz, y + dy, x + dx));
                const auto e = hn::Sub(p, q);
                acc = hn::MulAdd(e, e, acc);
            }
            hn::StoreU(acc, d, a + x);
        }
        for (; x < x_hi; ++x) {
            float acc = 0.f;
            for (int k = 0; k < L; ++k) {
                const float e = V[vol.at(z0 + k, y, x)] - V[vol.at(z0 + k + dz, y + dy, x + dx)];
                acc = std::fma(e, e, acc);
            }
            a[x] = acc;
        }
    }
}

// B[x] = sum over j < L (in order) of A[y0 + j][x], then
// C[x] = sum over i < L (in order) of B[x + i]: the distance of the cube at x.
static HWY_NOINLINE void Window(const float* A, int w, int L, int y0, int x_lo, int x_hi, float* B, float* C) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    int x = x_lo;
    for (; x + N <= x_hi; x += N) {
        auto acc = hn::LoadU(d, A + static_cast<std::size_t>(y0) * w + x);
        for (int j = 1; j < L; ++j) acc = hn::Add(acc, hn::LoadU(d, A + static_cast<std::size_t>(y0 + j) * w + x));
        hn::StoreU(acc, d, B + x);
    }
    for (; x < x_hi; ++x) {
        float acc = A[static_cast<std::size_t>(y0) * w + x];
        for (int j = 1; j < L; ++j) acc = acc + A[static_cast<std::size_t>(y0 + j) * w + x];
        B[x] = acc;
    }
    const int c_hi = x_hi - L + 1;
    x = x_lo;
    for (; x + N <= c_hi; x += N) {
        auto acc = hn::LoadU(d, B + x);
        for (int i = 1; i < L; ++i) acc = hn::Add(acc, hn::LoadU(d, B + x + i));
        hn::StoreU(acc, d, C + x);
    }
    for (; x < c_hi; ++x) {
        float acc = B[x];
        for (int i = 1; i < L; ++i) acc = acc + B[x + i];
        C[x] = acc;
    }
}

static HWY_NOINLINE void Match(const float* V, const Volume& vol, const Refs& refs, float limit, Matches& m, float* A,
                               float* B, float* C) {
    const int L = refs.L;
    const int R = refs.count();
    for (int r = 0; r < R; ++r) {
        const int iz = r / (refs.ny * refs.nx), iy = (r / refs.nx) % refs.ny, ix = r % refs.nx;
        m.count[r] = 0;
        m.insert(r, -1.f,
                 static_cast<std::int32_t>(vol.at(position(vol.t, L, refs.step, iz), position(vol.h, L, refs.step, iy),
                                                  position(vol.w, L, refs.step, ix))));
    }
    // All offsets per plane of references, so that the frames it reads stay
    // in cache (smaller bands of rows measured slower). Each reference still
    // sees its offsets in (dz, dy, dx) order.
    const int kBandRows = refs.ny;
    const int S = kBm4dSearch;
    for (int iz = 0; iz < refs.nz; ++iz) {
        const int z0 = position(vol.t, L, refs.step, iz);
        for (int band = 0; band < refs.ny; band += kBandRows) {
            const int band_end = std::min(refs.ny, band + kBandRows);
            const int row_lo = position(vol.h, L, refs.step, band);
            const int row_hi = position(vol.h, L, refs.step, band_end - 1) + L;
            for (int dz = -S; dz <= S; ++dz) {
                if (z0 + dz < 0 || z0 + dz > vol.t - L) continue;
                for (int dy = -S; dy <= S; ++dy) {
                    const int y_lo = std::max(row_lo, -dy), y_hi = std::min(row_hi, vol.h - dy);
                    if (y_hi - y_lo < L) continue;
                    for (int dx = -S; dx <= S; ++dx) {
                        if (dz == 0 && dy == 0 && dx == 0) continue;
                        const int x_lo = std::max(0, -dx), x_hi = std::min(vol.w, vol.w - dx);
                        if (x_hi - x_lo < L) continue;
                        Slab(V, vol, L, z0, dz, dy, dx, y_lo, y_hi, x_lo, x_hi, A);
                        for (int iy = band; iy < band_end; ++iy) {
                            const int y0 = position(vol.h, L, refs.step, iy);
                            if (y0 + dy < 0 || y0 + dy > vol.h - L) continue;
                            Window(A, vol.w, L, y0, x_lo, x_hi, B, C);
                            const int base = (iz * refs.ny + iy) * refs.nx;
                            for (int ix = 0; ix < refs.nx; ++ix) {
                                const int x0 = position(vol.w, L, refs.step, ix);
                                if (x0 + dx < 0 || x0 + dx > vol.w - L) continue;
                                const float s = C[x0];
                                if (s <= limit)
                                    m.insert(base + ix, s,
                                             static_cast<std::int32_t>(vol.at(z0 + dz, y0 + dy, x0 + dx)));
                            }
                        }
                    }
                }
            }
        }
    }
}

// One axis of the orthonormal 3D DCT on a group stored element-major
// (element (z, y, x) is a row of `stride` lanes, one per cube):
// dst[.., k, ..] = sum over i (in order) of M[k][i] src[.., i, ..], with M the
// DCT matrix or (inverse) its transpose. axis 0 is x, 1 y, 2 z.
static void DctAxis(const float* src, float* dst, int L, int axis, int stride, bool inverse) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const auto& D = transforms().dct[L == 5];
    const int unit = axis == 0 ? 1 : axis == 1 ? L : L * L;  // element step along the axis
    for (int a = 0; a < L; ++a) {
        for (int b = 0; b < L; ++b) {
            // the line through the other two coordinates (a, b)
            const int base = axis == 0 ? (a * L + b) * L : axis == 1 ? a * L * L + b : a * L + b;
            for (int k = 0; k < L; ++k) {
                float* out = dst + static_cast<std::size_t>(base + k * unit) * stride;
                for (int q = 0; q < stride; q += N) {
                    auto acc = hn::Mul(hn::Set(d, inverse ? D[0][k] : D[k][0]), hn::Load(d, src + static_cast<std::size_t>(base) * stride + q));
                    for (int i = 1; i < L; ++i)
                        acc = hn::MulAdd(hn::Set(d, inverse ? D[i][k] : D[k][i]),
                                         hn::Load(d, src + static_cast<std::size_t>(base + i * unit) * stride + q), acc);
                    hn::Store(acc, d, out + q);
                }
            }
        }
    }
}

// The 3D DCT (x, then y, then z) of an element-major group in s, using t as
// scratch; the result is in s.
static void Dct3(float* s, float* t, int L, int stride, bool inverse) {
    DctAxis(s, t, L, 0, stride, inverse);
    DctAxis(t, s, L, 1, stride, inverse);
    DctAxis(s, t, L, 2, stride, inverse);
    std::memcpy(s, t, static_cast<std::size_t>(L) * L * L * stride * sizeof(float));
}

// The orthonormal Haar transform along the group (m a power of two) of n-element
// vectors: pairs (s, d) = ((a + b) c, (a - b) c), c = 1/sqrt2, sums recursed.
static void Haar(float* g, int m, int n, bool inverse, float* tmp) {
    const hn::ScalableTag<float> dd;
    const int N = static_cast<int>(hn::Lanes(dd));
    const float c = 0.70710678118654752f;
    const auto vc = hn::Set(dd, c);
    auto pair = [&](const float* a, const float* b, float* plus, float* minus) {
        int e = 0;
        for (; e + N <= n; e += N) {
            const auto va = hn::LoadU(dd, a + e), vb = hn::LoadU(dd, b + e);
            hn::StoreU(hn::Mul(hn::Add(va, vb), vc), dd, plus + e);
            hn::StoreU(hn::Mul(hn::Sub(va, vb), vc), dd, minus + e);
        }
        for (; e < n; ++e) {
            plus[e] = (a[e] + b[e]) * c;
            minus[e] = (a[e] - b[e]) * c;
        }
    };
    if (!inverse) {
        for (int len = m; len > 1; len >>= 1) {
            for (int i = 0; i < len / 2; ++i)
                pair(g + static_cast<std::size_t>(2 * i) * n, g + static_cast<std::size_t>(2 * i + 1) * n,
                     tmp + static_cast<std::size_t>(i) * n, tmp + static_cast<std::size_t>(len / 2 + i) * n);
            std::memcpy(g, tmp, static_cast<std::size_t>(len) * n * sizeof(float));
        }
    } else {
        for (int len = 2; len <= m; len <<= 1) {
            for (int i = 0; i < len / 2; ++i)
                pair(g + static_cast<std::size_t>(i) * n, g + static_cast<std::size_t>(len / 2 + i) * n,
                     tmp + static_cast<std::size_t>(2 * i) * n, tmp + static_cast<std::size_t>(2 * i + 1) * n);
            std::memcpy(g, tmp, static_cast<std::size_t>(len) * n * sizeof(float));
        }
    }
}

struct StageBuffers {
    float* A;
    float* B;
    float* C;
    Matches matches;
    float* group;   // kMaxGroup cubes, cube-major
    float* bgroup;  // the basic estimate's
    float* tmp;
    float* soa;     // element-major copies: noisy, basic, scratch
    float* bsoa;
    float* tsoa;
};

// cube-major g (m cubes of n) <-> element-major s (n rows of stride lanes, zero padded)
static void ToElements(const float* g, int m, int n, int stride, float* s) {
    for (int e = 0; e < n; ++e) {
        float* row = s + static_cast<std::size_t>(e) * stride;
        for (int q = 0; q < m; ++q) row[q] = g[static_cast<std::size_t>(q) * n + e];
        for (int q = m; q < stride; ++q) row[q] = 0.f;
    }
}
static void ToCubes(const float* s, int m, int n, int stride, float* g) {
    for (int q = 0; q < m; ++q)
        for (int e = 0; e < n; ++e) g[static_cast<std::size_t>(q) * n + e] = s[static_cast<std::size_t>(e) * stride + q];
}

static HWY_NOINLINE void Stage(const float* noisy, const float* basic, const Volume& vol, float sigma,
                               const Bm4dStage& st, float* num, float* den, StageBuffers& buf) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const int L = st.cube, L2 = L * L, L3 = L2 * L;
    Refs refs{position_count(vol.t, L, st.step), position_count(vol.h, L, st.step), position_count(vol.w, L, st.step),
              L, st.step};
    const float sigma2 = sigma * sigma;
    const float limit = st.tau * static_cast<float>(L3) * sigma2;
    buf.matches.group = st.group;
    Match(basic ? basic : noisy, vol, refs, limit, buf.matches, buf.A, buf.B, buf.C);

    const float* kaiser = transforms().kaiser[L == 5];
    float window[kMaxCube3];
    for (int z = 0; z < L; ++z)
        for (int y = 0; y < L; ++y)
            for (int x = 0; x < L; ++x) window[(z * L + y) * L + x] = kaiser[z] * kaiser[y] * kaiser[x];
    const auto vthr = hn::Set(d, kBm4dLambda * sigma);
    const auto vsig2 = hn::Set(d, sigma2);
    const std::size_t plane = static_cast<std::size_t>(vol.h) * vol.w;

    auto gather = [&](const float* src, std::int32_t origin, float* cube) {
        for (int z = 0; z < L; ++z)
            for (int y = 0; y < L; ++y)
                std::memcpy(cube + (z * L + y) * L, src + origin + z * plane + static_cast<std::size_t>(y) * vol.w,
                            L * sizeof(float));
    };

    for (int r = 0; r < refs.count(); ++r) {
        int m = 1;
        while (2 * m <= buf.matches.count[r]) m *= 2;
        const int stride = (m + N - 1) / N * N;
        const std::int32_t* pos = buf.matches.pos + static_cast<std::size_t>(r) * st.group;
        // 4D transform: Haar along the group, then the 3D DCT of every cube.
        for (int q = 0; q < m; ++q) gather(noisy, pos[q], buf.group + static_cast<std::size_t>(q) * L3);
        Haar(buf.group, m, L3, false, buf.tmp);
        ToElements(buf.group, m, L3, stride, buf.soa);
        Dct3(buf.soa, buf.tsoa, L, stride, false);
        const std::size_t total = static_cast<std::size_t>(L3) * stride;
        float weight;
        if (!basic) {
            std::size_t kept = 0;
            for (std::size_t i = 0; i < total; i += N) {
                const auto v = hn::Load(d, buf.soa + i);
                const auto keep = hn::Ge(hn::Abs(v), vthr);
                kept += hn::CountTrue(d, keep);
                hn::Store(hn::IfThenElseZero(keep, v), d, buf.soa + i);
            }
            weight = 1.f / (sigma2 * static_cast<float>(std::max<std::size_t>(kept, 1)));
        } else {
            for (int q = 0; q < m; ++q) gather(basic, pos[q], buf.bgroup + static_cast<std::size_t>(q) * L3);
            Haar(buf.bgroup, m, L3, false, buf.tmp);
            ToElements(buf.bgroup, m, L3, stride, buf.bsoa);
            Dct3(buf.bsoa, buf.tsoa, L, stride, false);
            for (std::size_t i = 0; i < total; i += N) {
                const auto b = hn::Load(d, buf.bsoa + i);
                const auto b2 = hn::Mul(b, b);
                const auto wv = hn::Div(b2, hn::Add(b2, vsig2));
                hn::Store(hn::Mul(hn::Load(d, buf.soa + i), wv), d, buf.soa + i);
                hn::Store(wv, d, buf.bsoa + i);
            }
            // sum of W^2 over elements, then cubes, in order
            float energy = 0.f;
            for (int e = 0; e < L3; ++e) {
                const float* row = buf.bsoa + static_cast<std::size_t>(e) * stride;
                for (int q = 0; q < m; ++q) energy = std::fma(row[q], row[q], energy);
            }
            weight = 1.f / (sigma2 * std::max(energy, 1e-12f));
        }
        Dct3(buf.soa, buf.tsoa, L, stride, true);
        ToCubes(buf.soa, m, L3, stride, buf.group);
        Haar(buf.group, m, L3, true, buf.tmp);
        for (int q = 0; q < m; ++q) {
            const float* cube = buf.group + static_cast<std::size_t>(q) * L3;
            for (int z = 0; z < L; ++z)
                for (int y = 0; y < L; ++y) {
                    const std::size_t o = pos[q] + z * plane + static_cast<std::size_t>(y) * vol.w;
                    for (int x = 0; x < L; ++x) {
                        const float wv = weight * window[(z * L + y) * L + x];
                        num[o + x] = std::fma(wv, cube[(z * L + y) * L + x], num[o + x]);
                        den[o + x] = den[o + x] + wv;
                    }
                }
        }
    }
}

void Bm4dDenoiseVolume(const float* noisy, int t, int h, int w, float sigma, float* out, float* work,
                       std::size_t work_floats, float* basic_out) {
    if (t < 1 || h < 1 || w < 1 || !(sigma > 0.f)) throw std::invalid_argument("nss: BM4D kernel arguments are invalid");
    if (work_floats < bm4d_work_floats(t, h, w)) throw std::invalid_argument("nss: BM4D work buffer too small");
    const Volume vol{t, h, w};
    const std::size_t n = static_cast<std::size_t>(t) * h * w;
    // The buffer need not be aligned: start at its first 64-byte boundary
    // (bm4d_work_floats leaves room for it).
    float* p = work + ((64 - reinterpret_cast<std::uintptr_t>(work) % 64) % 64) / sizeof(float);
    auto take = [&p](std::size_t k) {
        float* r = p;
        p += align16(k);
        return r;
    };
    float* basic = basic_out ? basic_out : take(n);
    float* num = take(n);
    float* den = take(n);
    StageBuffers buf{};
    buf.A = take(static_cast<std::size_t>(h) * w);
    buf.B = take(static_cast<std::size_t>(w));
    buf.C = take(static_cast<std::size_t>(w));
    const int R = std::max(1, position_count(t, kBm4dHard.cube, kBm4dHard.step)) *
                  std::max(1, position_count(h, kBm4dHard.cube, kBm4dHard.step)) *
                  std::max(1, position_count(w, kBm4dHard.cube, kBm4dHard.step));
    buf.matches.dist = take(static_cast<std::size_t>(R) * kMaxGroup);
    buf.matches.pos = reinterpret_cast<std::int32_t*>(take(static_cast<std::size_t>(R) * kMaxGroup));
    buf.matches.count = reinterpret_cast<std::int32_t*>(take(static_cast<std::size_t>(R)));
    buf.group = take(static_cast<std::size_t>(kMaxGroup) * kMaxCube3);
    buf.bgroup = take(static_cast<std::size_t>(kMaxGroup) * kMaxCube3);
    buf.tmp = take(static_cast<std::size_t>(kMaxGroup) * kMaxCube3);
    buf.soa = take(static_cast<std::size_t>(kMaxGroup + 16) * kMaxCube3);
    buf.bsoa = take(static_cast<std::size_t>(kMaxGroup + 16) * kMaxCube3);
    buf.tsoa = take(static_cast<std::size_t>(kMaxGroup + 16) * kMaxCube3);

    // A volume smaller than a stage's cube along an axis: that stage is skipped
    // (the hard stage's estimate is the noisy volume, the final the basic).
    auto fits = [&](const Bm4dStage& st) { return t >= st.cube && h >= st.cube && w >= st.cube; };
    if (fits(kBm4dHard)) {
        std::fill(num, num + n, 0.f);
        std::fill(den, den + n, 0.f);
        Stage(noisy, nullptr, vol, sigma, kBm4dHard, num, den, buf);
        for (std::size_t i = 0; i < n; ++i) basic[i] = num[i] / den[i];
    } else {
        std::memcpy(basic, noisy, n * sizeof(float));
    }
    if (fits(kBm4dWiener)) {
        std::fill(num, num + n, 0.f);
        std::fill(den, den + n, 0.f);
        Stage(noisy, basic, vol, sigma, kBm4dWiener, num, den, buf);
        for (std::size_t i = 0; i < n; ++i) out[i] = num[i] / den[i];
    } else {
        std::memcpy(out, basic, n * sizeof(float));
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(Bm4dDenoiseVolume);

std::size_t bm4d_work_floats(int t, int h, int w) {
    using namespace bm4d_detail;
    const std::size_t n = static_cast<std::size_t>(t) * h * w;
    const std::size_t R = static_cast<std::size_t>(std::max(1, position_count(t, kBm4dHard.cube, kBm4dHard.step))) *
                          std::max(1, position_count(h, kBm4dHard.cube, kBm4dHard.step)) *
                          std::max(1, position_count(w, kBm4dHard.cube, kBm4dHard.step));
    return 3 * align16(n) + align16(static_cast<std::size_t>(h) * w) + 2 * align16(static_cast<std::size_t>(w)) +
           2 * align16(R * kMaxGroup) + align16(R) + 3 * align16(static_cast<std::size_t>(kMaxGroup) * kMaxCube3) +
           3 * align16(static_cast<std::size_t>(kMaxGroup + 16) * kMaxCube3) + 16;
}

void bm4d_denoise_volume(const float* noisy, int t, int h, int w, float sigma, float* out, float* work,
                         std::size_t work_floats, float* basic) {
    HWY_DYNAMIC_DISPATCH(Bm4dDenoiseVolume)(noisy, t, h, w, sigma, out, work, work_floats, basic);
}
}  // namespace nss
#endif
