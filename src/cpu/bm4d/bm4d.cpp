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
#include <initializer_list>
#include <stdexcept>
#include <utility>

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
constexpr int kCubeStride = (kMaxCube3 + 15) & ~15;  // floats a cube in a group buffer (zero padded)
constexpr int kWindowPad = 64;  // floats past the window row B (RefSums' loads)
constexpr int kSlabBatch = 8;   // offsets a Slabs pass
constexpr int kBandRows = 8;    // rows of references a band

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
    float* worst;  // dist[group - 1] once full, else +inf: what a new distance must beat
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
        if (c + 1 == group) worst[r] = dr[group - 1];
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
template <int L>
static HWY_NOINLINE void Slab(const float* V, const Volume& vol, int z0, int dz, int dy, int dx, int y_lo, int y_hi,
                              int x_lo, int x_hi, float* A) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    for (int y = y_lo; y < y_hi; ++y) {
        const float* p[L];
        const float* q[L];
        for (int k = 0; k < L; ++k) {
            p[k] = V + vol.at(z0 + k, y, 0);
            q[k] = V + vol.at(z0 + k + dz, y + dy, 0) + dx;
        }
        float* a = A + static_cast<std::size_t>(y) * vol.w;
        int x = x_lo;
        for (; x + N <= x_hi; x += N) {
            auto acc = hn::Zero(d);
            for (int k = 0; k < L; ++k) {
                const auto e = hn::Sub(hn::LoadU(d, p[k] + x), hn::LoadU(d, q[k] + x));
                acc = hn::MulAdd(e, e, acc);
            }
            hn::StoreU(acc, d, a + x);
        }
        for (; x < x_hi; ++x) {
            float acc = 0.f;
            for (int k = 0; k < L; ++k) {
                const float e = p[k][x] - q[k][x];
                acc = std::fma(e, e, acc);
            }
            a[x] = acc;
        }
    }
}

// Slab for the D offsets dx0, dx0 + 1, ... at once (the reference frames'
// loads shared): offset j's A at A + j h w. A run of columns ends with a
// vector over its last N columns (columns done twice get the same values).
template <int L, int D>
static HWY_NOINLINE void Slabs(const float* V, const Volume& vol, int z0, int dz, int dy, int dx0, int y_lo, int y_hi,
                               float* A) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const std::size_t hw = static_cast<std::size_t>(vol.h) * vol.w;
    // the columns every offset has, a vector at a time for all of them; each
    // offset's other columns on its own
    const int c_lo = std::max(0, -dx0), c_hi = std::min(vol.w, vol.w - (dx0 + D - 1));
    const bool common = c_hi - c_lo >= N;
    for (int y = y_lo; y < y_hi; ++y) {
        const float* p[L];
        const float* q[L];
        for (int k = 0; k < L; ++k) {
            p[k] = V + vol.at(z0 + k, y, 0);
            q[k] = V + vol.at(z0 + k + dz, y + dy, 0) + dx0;
        }
        float* a = A + static_cast<std::size_t>(y) * vol.w;
        auto all = [&](int x) {
            hn::Vec<decltype(d)> acc[D];
            for (int j = 0; j < D; ++j) acc[j] = hn::Zero(d);
            for (int k = 0; k < L; ++k) {
                const auto pv = hn::LoadU(d, p[k] + x);
                for (int j = 0; j < D; ++j) {
                    const auto e = hn::Sub(pv, hn::LoadU(d, q[k] + x + j));
                    acc[j] = hn::MulAdd(e, e, acc[j]);
                }
            }
            for (int j = 0; j < D; ++j) hn::StoreU(acc[j], d, a + j * hw + x);
        };
        if (common) {
            int x = c_lo;
            for (; x + N <= c_hi; x += N) all(x);
            if (x < c_hi) all(c_hi - N);
        }
        for (int j = 0; j < D; ++j) {
            const int lo = std::max(0, -(dx0 + j)), hi = std::min(vol.w, vol.w - (dx0 + j));
            auto one = [&](int x) {
                auto acc = hn::Zero(d);
                for (int k = 0; k < L; ++k) {
                    const auto e = hn::Sub(hn::LoadU(d, p[k] + x), hn::LoadU(d, q[k] + x + j));
                    acc = hn::MulAdd(e, e, acc);
                }
                hn::StoreU(acc, d, a + j * hw + x);
            };
            auto single = [&](int xx) {
                float acc = 0.f;
                for (int k = 0; k < L; ++k) {
                    const float e = p[k][xx] - q[k][xx + j];
                    acc = std::fma(e, e, acc);
                }
                a[j * hw + xx] = acc;
            };
            // [lo, hi) less the common run
            auto run = [&](int s0, int s1) {
                if (s1 <= s0) return;
                if (hi - lo >= N) {
                    int x = s0;
                    for (; x + N <= s1; x += N) one(x);
                    if (x < s1) one(std::max(lo, s1 - N));
                } else {
                    for (int xx = s0; xx < s1; ++xx) single(xx);
                }
            };
            if (common) {
                run(lo, std::min(c_lo, hi));
                run(std::max(c_hi, lo), hi);
            } else {
                run(lo, hi);
            }
        }
    }
}

// B[x] = sum over j < L (in order) of A[y0 + j][x].
template <int L>
static HWY_NOINLINE void Window(const float* A, int w, int y0, int x_lo, int x_hi, float* B) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const float* rows[L];
    for (int j = 0; j < L; ++j) rows[j] = A + static_cast<std::size_t>(y0 + j) * w;
    auto vec = [&](int x) {
        auto acc = hn::LoadU(d, rows[0] + x);
        for (int j = 1; j < L; ++j) acc = hn::Add(acc, hn::LoadU(d, rows[j] + x));
        hn::StoreU(acc, d, B + x);
    };
    int x = x_lo;
    for (; x + N <= x_hi; x += N) vec(x);
    if (x < x_hi && x_hi - x_lo >= N) {
        vec(x_hi - N);  // the last N columns (columns done twice get the same values)
        x = x_hi;
    }
    for (; x < x_hi; ++x) {
        float acc = rows[0][x];
        for (int j = 1; j < L; ++j) acc = acc + rows[j][x];
        B[x] = acc;
    }
}

// Cr[ix] = sum over i < L (in order) of B[x0 + i], x0 the reference ix's
// column: the distance of its candidate, for ix in [lo, hi). The references
// at 3 ix (step 3, all but perhaps the last) a vector at a time, by
// deinterleaved loads (B has kWindowPad floats after the row).
template <int L>
static HWY_NOINLINE void RefSums(const float* B, int w, int nx, int step, int lo, int hi, float* Cr) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    static_assert(L <= 6, "two deinterleaved loads");
    int ix = lo;
    if (step == 3) {
        const int regular = std::min(hi, (w - L) / 3 + 1);  // the references at 3 ix
        auto vec = [&](int k) {
            hn::Vec<decltype(d)> b0, b1, b2, c0, c1, c2;
            hn::LoadInterleaved3(d, B + 3 * k, b0, b1, b2);
            hn::LoadInterleaved3(d, B + 3 * k + 3, c0, c1, c2);
            auto acc = hn::Add(hn::Add(b0, b1), b2);
            if (L > 3) acc = hn::Add(acc, c0);
            if (L > 4) acc = hn::Add(acc, c1);
            if (L > 5) acc = hn::Add(acc, c2);
            hn::StoreU(acc, d, Cr + k);
        };
        for (; ix + N <= regular; ix += N) vec(ix);
        if (ix < regular && regular - lo >= N) {
            vec(regular - N);  // the last N (done twice: the same values)
            ix = regular;
        }
    }
    for (; ix < hi; ++ix) {
        const int x0 = position(w, L, step, ix);
        float acc = B[x0];
        for (int i = 1; i < L; ++i) acc = acc + B[x0 + i];
        Cr[ix] = acc;
    }
    (void)nx;
}

template <int L>
static HWY_NOINLINE void Match(const float* V, const Volume& vol, const Refs& refs, float limit, Matches& m, float* A,
                               float* B, float* Cr) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const auto vlimit = hn::Set(d, limit);
    const int R = refs.count();
    for (int r = 0; r < R; ++r) {
        const int iz = r / (refs.ny * refs.nx), iy = (r / refs.nx) % refs.ny, ix = r % refs.nx;
        m.count[r] = 0;
        m.worst[r] = HUGE_VALF;
        m.insert(r, -1.f,
                 static_cast<std::int32_t>(vol.at(position(vol.t, L, refs.step, iz), position(vol.h, L, refs.step, iy),
                                                  position(vol.w, L, refs.step, ix))));
    }
    // Per plane of references, per band of kBandRows of their rows (the
    // frames' rows they read stay in cache), every offset; kSlabBatch
    // neighbouring dx at a time. Each reference still sees its offsets in
    // (dz, dy, dx) order.
    const int S = kBm4dSearch;
    const std::size_t hw = static_cast<std::size_t>(vol.h) * vol.w;
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
                    for (int dx0 = -S; dx0 <= S; dx0 += kSlabBatch) {
                        const int D = std::min(kSlabBatch, S - dx0 + 1);
                        switch (D) {
                            case 8: Slabs<L, 8>(V, vol, z0, dz, dy, dx0, y_lo, y_hi, A); break;
                            case 7: Slabs<L, 7>(V, vol, z0, dz, dy, dx0, y_lo, y_hi, A); break;
                            case 6: Slabs<L, 6>(V, vol, z0, dz, dy, dx0, y_lo, y_hi, A); break;
                            case 5: Slabs<L, 5>(V, vol, z0, dz, dy, dx0, y_lo, y_hi, A); break;
                            case 4: Slabs<L, 4>(V, vol, z0, dz, dy, dx0, y_lo, y_hi, A); break;
                            case 3: Slabs<L, 3>(V, vol, z0, dz, dy, dx0, y_lo, y_hi, A); break;
                            case 2: Slabs<L, 2>(V, vol, z0, dz, dy, dx0, y_lo, y_hi, A); break;
                            default: Slabs<L, 1>(V, vol, z0, dz, dy, dx0, y_lo, y_hi, A); break;
                        }
                        for (int j = 0; j < D; ++j) {
                            const int dx = dx0 + j;
                            if (dz == 0 && dy == 0 && dx == 0) continue;
                            const int x_lo = std::max(0, -dx), x_hi = std::min(vol.w, vol.w - dx);
                            if (x_hi - x_lo < L) continue;
                            const float* Aj = A + j * hw;
                            for (int iy = band; iy < band_end; ++iy) {
                                const int y0 = position(vol.h, L, refs.step, iy);
                                if (y0 + dy < 0 || y0 + dy > vol.h - L) continue;
                                Window<L>(Aj, vol.w, y0, x_lo, x_hi, B);
                                const int base = (iz * refs.ny + iy) * refs.nx;
                                // the references whose candidate exists: ix in [ix_lo, ix_hi)
                                const float* worst = m.worst + base;
                                int ix_lo = 0, ix_hi = refs.nx;
                                while (ix_lo < ix_hi && position(vol.w, L, refs.step, ix_lo) + dx < 0) ++ix_lo;
                                while (ix_hi > ix_lo && position(vol.w, L, refs.step, ix_hi - 1) + dx > vol.w - L) --ix_hi;
                                RefSums<L>(B, vol.w, refs.nx, refs.step, ix_lo, ix_hi, Cr);
                                // as insert's test of a full list, without reading it, a vector at a time
                                int ix = ix_lo;
                                for (; ix + N <= ix_hi; ix += N) {
                                    const auto sv = hn::LoadU(d, Cr + ix);
                                    const auto hit = hn::And(hn::Le(sv, vlimit), hn::Lt(sv, hn::LoadU(d, worst + ix)));
                                    if (hn::AllFalse(d, hit)) continue;
                                    for (int k = ix; k < ix + N; ++k) {
                                        const float sk = Cr[k];
                                        if (sk <= limit && sk < worst[k])
                                            m.insert(base + k, sk,
                                                     static_cast<std::int32_t>(vol.at(z0 + dz, y0 + dy, position(vol.w, L, refs.step, k) + dx)));
                                    }
                                }
                                if (ix < ix_hi && ix_hi - ix_lo >= N) {
                                    // the last N, less the lanes already tested
                                    const int k0 = ix_hi - N;
                                    const auto sv = hn::LoadU(d, Cr + k0);
                                    const auto hit = hn::AndNot(hn::FirstN(d, static_cast<std::size_t>(ix - k0)),
                                                                hn::And(hn::Le(sv, vlimit), hn::Lt(sv, hn::LoadU(d, worst + k0))));
                                    if (!hn::AllFalse(d, hit))
                                        for (int k = ix; k < ix_hi; ++k) {
                                            const float sk = Cr[k];
                                            if (sk <= limit && sk < worst[k])
                                                m.insert(base + k, sk,
                                                         static_cast<std::int32_t>(vol.at(z0 + dz, y0 + dy, position(vol.w, L, refs.step, k) + dx)));
                                        }
                                    ix = ix_hi;
                                }
                                for (; ix < ix_hi; ++ix) {
                                    const float sk = Cr[ix];
                                    if (sk <= limit && sk < worst[ix])
                                        m.insert(base + ix, sk,
                                                 static_cast<std::int32_t>(vol.at(z0 + dz, y0 + dy, position(vol.w, L, refs.step, ix) + dx)));
                                }
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
// DCT matrix or (inverse) its transpose. axis 0 is x, 1 y, 2 z. A line's L
// inputs are loaded once for its L outputs.
template <int L, int axis>
static void DctAxis(const float* src, float* dst, int stride, bool inverse) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const auto& D = transforms().dct[L == 5];
    float M[L][L];
    for (int k = 0; k < L; ++k)
        for (int i = 0; i < L; ++i) M[k][i] = inverse ? D[i][k] : D[k][i];
    constexpr int unit = axis == 0 ? 1 : axis == 1 ? L : L * L;  // element step along the axis
    for (int a = 0; a < L; ++a) {
        for (int b = 0; b < L; ++b) {
            // the line through the other two coordinates (a, b)
            const int base = axis == 0 ? (a * L + b) * L : axis == 1 ? a * L * L + b : a * L + b;
            const float* in = src + static_cast<std::size_t>(base) * stride;
            float* out = dst + static_cast<std::size_t>(base) * stride;
            const std::size_t step = static_cast<std::size_t>(unit) * stride;
            for (int q = 0; q < stride; q += N) {
                hn::Vec<decltype(d)> v[L];
                for (int i = 0; i < L; ++i) v[i] = hn::Load(d, in + i * step + q);
                for (int k = 0; k < L; ++k) {
                    auto acc = hn::Mul(hn::Set(d, M[k][0]), v[0]);
                    for (int i = 1; i < L; ++i) acc = hn::MulAdd(hn::Set(d, M[k][i]), v[i], acc);
                    hn::Store(acc, d, out + k * step + q);
                }
            }
        }
    }
}

// The 3D DCT (x, then y, then z) of an element-major group in s, using t as
// scratch; the result is in s (the two pointers swap).
static void Dct3(float*& s, float*& t, int L, int stride, bool inverse) {
    if (L == 4) {
        DctAxis<4, 0>(s, t, stride, inverse);
        DctAxis<4, 1>(t, s, stride, inverse);
        DctAxis<4, 2>(s, t, stride, inverse);
    } else {
        DctAxis<5, 0>(s, t, stride, inverse);
        DctAxis<5, 1>(t, s, stride, inverse);
        DctAxis<5, 2>(s, t, stride, inverse);
    }
    std::swap(s, t);
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
        // level len reads its rows from src (g, then tmp and g in turn); the
        // differences go to their final rows len / 2 + i of g, the sums to the
        // other buffer, the pairs from the last, so a row of g is read before
        // it is written
        float* src = g;
        for (int len = m; len > 1; len >>= 1) {
            float* sums = src == g ? tmp : g;
            for (int i = len / 2 - 1; i >= 0; --i)
                pair(src + static_cast<std::size_t>(2 * i) * n, src + static_cast<std::size_t>(2 * i + 1) * n,
                     sums + static_cast<std::size_t>(i) * n, g + static_cast<std::size_t>(len / 2 + i) * n);
            src = sums;
        }
        if (src != g) std::memcpy(g, src, static_cast<std::size_t>(n) * sizeof(float));
    } else {
        // level len: the sums from src, the differences from g; out to the
        // other buffer, the last level to g (the first, len 2, may be in place)
        int levels = 0;
        while ((1 << levels) < m) ++levels;
        float* src = g;
        for (int len = 2, t = 1; len <= m; len <<= 1, ++t) {
            float* dst = (levels - t) % 2 == 0 ? g : tmp;
            for (int i = 0; i < len / 2; ++i)
                pair(src + static_cast<std::size_t>(i) * n, g + static_cast<std::size_t>(len / 2 + i) * n,
                     dst + static_cast<std::size_t>(2 * i) * n, dst + static_cast<std::size_t>(2 * i + 1) * n);
            src = dst;
        }
    }
}

// out[i] = bm4d_fixed(a[i] * b[i]): x 2^32 (exact), widened to double (exact)
// and rounded to an integer, ties to even, by adding 1.5 2^52 (|value| < 2^51),
// whose bits less the constant's are the integer; a vector with a lane outside
// that range (or NaN) takes llrint.
static HWY_INLINE void FixedProducts(const float* a, const float* b, int n, std::int64_t* out) {
    const hn::ScalableTag<double> dd;
    const hn::Rebind<float, decltype(dd)> df;
    const hn::RebindToSigned<decltype(dd)> di;
    const int N = static_cast<int>(hn::Lanes(dd));
    const auto scale = hn::Set(df, kBm4dFixedScale);
    const auto bound = hn::Set(df, 2251799813685248.f);  // 2^51
    const auto magic = hn::Set(dd, 6755399441055744.0);  // 1.5 2^52
    const auto magic_bits = hn::BitCast(di, magic);
    int i = 0;
    for (; i + N <= n; i += N) {
        const auto f = hn::Mul(hn::Mul(hn::LoadU(df, a + i), hn::LoadU(df, b + i)), scale);
        if (hn::AllTrue(df, hn::Lt(hn::Abs(f), bound))) {
            const auto r = hn::Add(hn::PromoteTo(dd, f), magic);
            hn::StoreU(hn::Sub(hn::BitCast(di, r), magic_bits), di, out + i);
        } else {
            for (int j = i; j < i + N; ++j) out[j] = bm4d_fixed(a[j] * b[j]);
        }
    }
    for (; i < n; ++i) out[i] = bm4d_fixed(a[i] * b[i]);
}

struct StageBuffers {
    float* A;
    float* B;   // kWindowPad floats past the row
    float* Cr;  // the distances of a row's references
    Matches matches;
    float* group;   // kMaxGroup cubes, cube-major
    float* bgroup;  // the basic estimate's
    float* tmp;
    float* soa;     // element-major copies: noisy, basic, scratch
    float* bsoa;
    float* tsoa;
};

// dst[c ds + r] = src[r ss + c] for r < rows, c < cols: 8 x 8 blocks (4 x 4
// with 128-bit vectors), the rest one at a time.
static void Transpose(const float* src, std::size_t ss, float* dst, std::size_t ds, int rows, int cols) {
    int r0 = 0, c_end = 0;
#if HWY_TARGET != HWY_SCALAR
#if HWY_MAX_BYTES >= 32
    constexpr int kB = 8;
    const hn::FixedTag<float, 8> d8;
    const hn::Repartition<uint64_t, decltype(d8)> d64;
    const auto il = [&](hn::Vec<decltype(d8)> a, hn::Vec<decltype(d8)> b) {
        return hn::BitCast(d8, hn::InterleaveLower(d64, hn::BitCast(d64, a), hn::BitCast(d64, b)));
    };
    const auto iu = [&](hn::Vec<decltype(d8)> a, hn::Vec<decltype(d8)> b) {
        return hn::BitCast(d8, hn::InterleaveUpper(d64, hn::BitCast(d64, a), hn::BitCast(d64, b)));
    };
    for (; r0 + kB <= rows; r0 += kB) {
        const float* sr = src + r0 * ss;
        int c0 = 0;
        for (; c0 + kB <= cols; c0 += kB) {
            const auto a0 = hn::LoadU(d8, sr + 0 * ss + c0), a1 = hn::LoadU(d8, sr + 1 * ss + c0);
            const auto a2 = hn::LoadU(d8, sr + 2 * ss + c0), a3 = hn::LoadU(d8, sr + 3 * ss + c0);
            const auto a4 = hn::LoadU(d8, sr + 4 * ss + c0), a5 = hn::LoadU(d8, sr + 5 * ss + c0);
            const auto a6 = hn::LoadU(d8, sr + 6 * ss + c0), a7 = hn::LoadU(d8, sr + 7 * ss + c0);
            const auto t0 = hn::InterleaveLower(d8, a0, a1), t1 = hn::InterleaveUpper(d8, a0, a1);
            const auto t2 = hn::InterleaveLower(d8, a2, a3), t3 = hn::InterleaveUpper(d8, a2, a3);
            const auto t4 = hn::InterleaveLower(d8, a4, a5), t5 = hn::InterleaveUpper(d8, a4, a5);
            const auto t6 = hn::InterleaveLower(d8, a6, a7), t7 = hn::InterleaveUpper(d8, a6, a7);
            const auto u0 = il(t0, t2), u1 = iu(t0, t2), u2 = il(t1, t3), u3 = iu(t1, t3);
            const auto u4 = il(t4, t6), u5 = iu(t4, t6), u6 = il(t5, t7), u7 = iu(t5, t7);
            float* o = dst + c0 * ds + r0;
            hn::StoreU(hn::ConcatLowerLower(d8, u4, u0), d8, o + 0 * ds);
            hn::StoreU(hn::ConcatLowerLower(d8, u5, u1), d8, o + 1 * ds);
            hn::StoreU(hn::ConcatLowerLower(d8, u6, u2), d8, o + 2 * ds);
            hn::StoreU(hn::ConcatLowerLower(d8, u7, u3), d8, o + 3 * ds);
            hn::StoreU(hn::ConcatUpperUpper(d8, u4, u0), d8, o + 4 * ds);
            hn::StoreU(hn::ConcatUpperUpper(d8, u5, u1), d8, o + 5 * ds);
            hn::StoreU(hn::ConcatUpperUpper(d8, u6, u2), d8, o + 6 * ds);
            hn::StoreU(hn::ConcatUpperUpper(d8, u7, u3), d8, o + 7 * ds);
        }
        c_end = c0;
    }
#else
    constexpr int kB = 4;
    const hn::FixedTag<float, 4> d4;
    for (; r0 + kB <= rows; r0 += kB) {
        const float* sr = src + r0 * ss;
        int c0 = 0;
        for (; c0 + kB <= cols; c0 += kB) {
            const auto a0 = hn::LoadU(d4, sr + 0 * ss + c0), a1 = hn::LoadU(d4, sr + 1 * ss + c0);
            const auto a2 = hn::LoadU(d4, sr + 2 * ss + c0), a3 = hn::LoadU(d4, sr + 3 * ss + c0);
            const auto t0 = hn::InterleaveLower(d4, a0, a1), t1 = hn::InterleaveLower(d4, a2, a3);
            const auto t2 = hn::InterleaveUpper(d4, a0, a1), t3 = hn::InterleaveUpper(d4, a2, a3);
            float* o = dst + c0 * ds + r0;
            hn::StoreU(hn::ConcatLowerLower(d4, t1, t0), d4, o + 0 * ds);
            hn::StoreU(hn::ConcatUpperUpper(d4, t1, t0), d4, o + 1 * ds);
            hn::StoreU(hn::ConcatLowerLower(d4, t3, t2), d4, o + 2 * ds);
            hn::StoreU(hn::ConcatUpperUpper(d4, t3, t2), d4, o + 3 * ds);
        }
        c_end = c0;
    }
#endif
#endif
    // the blocks' rows past c_end, then the rows past the blocks
    for (int r = 0; r < r0; ++r)
        for (int c = c_end; c < cols; ++c) dst[c * ds + r] = src[r * ss + c];
    for (int r = r0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) dst[c * ds + r] = src[r * ss + c];
}

// cube-major g (m cubes of n at a stride of cs) <-> element-major s (n rows of
// stride lanes, zero padded)
static void ToElements(const float* g, int m, int n, int cs, int stride, float* s) {
    Transpose(g, static_cast<std::size_t>(cs), s, static_cast<std::size_t>(stride), m, n);
    if (m < stride)
        for (int e = 0; e < n; ++e)
            for (int q = m; q < stride; ++q) s[static_cast<std::size_t>(e) * stride + q] = 0.f;
}
static void ToCubes(const float* s, int m, int n, int stride, int cs, float* g) {
    Transpose(s, static_cast<std::size_t>(stride), g, static_cast<std::size_t>(cs), n, m);
}

template <int L>
static HWY_NOINLINE void Stage(const float* noisy, const float* basic, const Volume& vol, float sigma,
                               const Bm4dStage& st, std::int64_t* num, std::int64_t* den, StageBuffers& buf) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    constexpr int L2 = L * L, L3 = L2 * L;
    constexpr int cs = (L3 + 15) & ~15;  // a cube's floats in the group buffers
    Refs refs{position_count(vol.t, L, st.step), position_count(vol.h, L, st.step), position_count(vol.w, L, st.step),
              L, st.step};
    const float sigma2 = sigma * sigma;
    const float limit = st.tau * static_cast<float>(L3) * sigma2;
    buf.matches.group = st.group;
    Match<L>(basic ? basic : noisy, vol, refs, limit, buf.matches, buf.A, buf.B, buf.Cr);

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
        for (int q = 0; q < m; ++q) gather(noisy, pos[q], buf.group + static_cast<std::size_t>(q) * cs);
        Haar(buf.group, m, cs, false, buf.tmp);
        ToElements(buf.group, m, L3, cs, stride, buf.soa);
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
            weight = 1.f / static_cast<float>(std::max<std::size_t>(kept, 1));
        } else {
            for (int q = 0; q < m; ++q) gather(basic, pos[q], buf.bgroup + static_cast<std::size_t>(q) * cs);
            Haar(buf.bgroup, m, cs, false, buf.tmp);
            ToElements(buf.bgroup, m, L3, cs, stride, buf.bsoa);
            Dct3(buf.bsoa, buf.tsoa, L, stride, false);
            for (std::size_t i = 0; i < total; i += N) {
                const auto b = hn::Load(d, buf.bsoa + i);
                const auto b2 = hn::Mul(b, b);
                const auto wv = hn::Div(b2, hn::Add(b2, vsig2));
                hn::Store(hn::Mul(hn::Load(d, buf.soa + i), wv), d, buf.soa + i);
                hn::Store(wv, d, buf.bsoa + i);
            }
            // sum of W^2: the 32-lane sum over t = e m + q (elements, then cubes);
            // m divides 32, so the terms of element e go to the lanes
            // (e m) % 32 + q in order
            float lane[32] = {};
            if (m >= N) {
                hn::Vec<decltype(d)> acc[32];
                for (int k = 0; k < 32 / N; ++k) acc[k] = hn::Zero(d);
                for (int e = 0; e < L3; ++e) {
                    const float* row = buf.bsoa + static_cast<std::size_t>(e) * stride;
                    const int base = (e * m) & 31;
                    for (int q = 0; q < m; q += N) {
                        const auto wv = hn::Load(d, row + q);
                        acc[(base + q) / N] = hn::MulAdd(wv, wv, acc[(base + q) / N]);
                    }
                }
                for (int k = 0; k < 32 / N; ++k) hn::StoreU(acc[k], d, lane + k * N);
            } else {
                for (int e = 0; e < L3; ++e) {
                    const float* row = buf.bsoa + static_cast<std::size_t>(e) * stride;
                    const int base = (e * m) & 31;
                    for (int q = 0; q < m; ++q) lane[base + q] = std::fma(row[q], row[q], lane[base + q]);
                }
            }
            for (int half = 16; half > 0; half >>= 1)
                for (int l = 0; l < half; ++l) lane[l] = lane[l] + lane[l + half];
            const float energy = lane[0];
            weight = 1.f / std::max(energy, kBm4dEnergyFloor);
        }
        Dct3(buf.soa, buf.tsoa, L, stride, true);
        ToCubes(buf.soa, m, L3, stride, cs, buf.group);
        Haar(buf.group, m, cs, true, buf.tmp);
        // the weights wv of the elements and their fixed-point terms are the
        // group's; the numerators' terms a cube at a time
        float wv[kMaxCube3];
        std::int64_t fden[kMaxCube3], fnum[kMaxCube3];
        for (int e = 0; e < L3; ++e) {
            wv[e] = weight * window[e];
            fden[e] = bm4d_fixed(wv[e]);
        }
        for (int q = 0; q < m; ++q) {
            FixedProducts(wv, buf.group + static_cast<std::size_t>(q) * cs, L3, fnum);
            for (int z = 0; z < L; ++z)
                for (int y = 0; y < L; ++y) {
                    const std::size_t o = pos[q] + z * plane + static_cast<std::size_t>(y) * vol.w;
                    const int e0 = (z * L + y) * L;
                    for (int x = 0; x < L; ++x) {
                        num[o + x] += fnum[e0 + x];
                        den[o + x] += fden[e0 + x];
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
    auto* num = reinterpret_cast<std::int64_t*>(take(2 * n));
    auto* den = reinterpret_cast<std::int64_t*>(take(2 * n));
    StageBuffers buf{};
    buf.A = take(static_cast<std::size_t>(kSlabBatch) * h * w);
    buf.B = take(static_cast<std::size_t>(w) + kWindowPad);
    buf.Cr = take(static_cast<std::size_t>(w));
    const int R = std::max(1, position_count(t, kBm4dHard.cube, kBm4dHard.step)) *
                  std::max(1, position_count(h, kBm4dHard.cube, kBm4dHard.step)) *
                  std::max(1, position_count(w, kBm4dHard.cube, kBm4dHard.step));
    buf.matches.dist = take(static_cast<std::size_t>(R) * kMaxGroup);
    buf.matches.pos = reinterpret_cast<std::int32_t*>(take(static_cast<std::size_t>(R) * kMaxGroup));
    buf.matches.count = reinterpret_cast<std::int32_t*>(take(static_cast<std::size_t>(R)));
    buf.matches.worst = take(static_cast<std::size_t>(R));
    buf.group = take(static_cast<std::size_t>(kMaxGroup) * kCubeStride);
    buf.bgroup = take(static_cast<std::size_t>(kMaxGroup) * kCubeStride);
    buf.tmp = take(static_cast<std::size_t>(kMaxGroup) * kCubeStride);
    // the cubes' padding stays zero (the gathers do not write it, Haar keeps zeros)
    for (float* b : {buf.group, buf.bgroup, buf.tmp}) std::fill(b, b + static_cast<std::size_t>(kMaxGroup) * kCubeStride, 0.f);
    buf.soa = take(static_cast<std::size_t>(kMaxGroup + 16) * kMaxCube3);
    buf.bsoa = take(static_cast<std::size_t>(kMaxGroup + 16) * kMaxCube3);
    buf.tsoa = take(static_cast<std::size_t>(kMaxGroup + 16) * kMaxCube3);

    // A volume smaller than a stage's cube along an axis: that stage is skipped
    // (the hard stage's estimate is the noisy volume, the final the basic).
    auto fits = [&](const Bm4dStage& st) { return t >= st.cube && h >= st.cube && w >= st.cube; };
    if (fits(kBm4dHard)) {
        std::fill(num, num + n, std::int64_t{0});
        std::fill(den, den + n, std::int64_t{0});
        Stage<kBm4dHard.cube>(noisy, nullptr, vol, sigma, kBm4dHard, num, den, buf);
        for (std::size_t i = 0; i < n; ++i) basic[i] = bm4d_ratio(num[i], den[i]);
    } else {
        std::memcpy(basic, noisy, n * sizeof(float));
    }
    if (fits(kBm4dWiener)) {
        std::fill(num, num + n, std::int64_t{0});
        std::fill(den, den + n, std::int64_t{0});
        Stage<kBm4dWiener.cube>(noisy, basic, vol, sigma, kBm4dWiener, num, den, buf);
        for (std::size_t i = 0; i < n; ++i) out[i] = bm4d_ratio(num[i], den[i]);
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
    return align16(n) + 2 * align16(2 * n) + align16(static_cast<std::size_t>(kSlabBatch) * h * w) + align16(static_cast<std::size_t>(w) + kWindowPad) + align16(static_cast<std::size_t>(w)) +
           2 * align16(R * kMaxGroup) + 2 * align16(R) + 3 * align16(static_cast<std::size_t>(kMaxGroup) * kCubeStride) +
           3 * align16(static_cast<std::size_t>(kMaxGroup + 16) * kMaxCube3) + 16;
}

void bm4d_denoise_volume(const float* noisy, int t, int h, int w, float sigma, float* out, float* work,
                         std::size_t work_floats, float* basic) {
    HWY_DYNAMIC_DISPATCH(Bm4dDenoiseVolume)(noisy, t, h, w, sigma, out, work, work_floats, basic);
}
}  // namespace nss
#endif
