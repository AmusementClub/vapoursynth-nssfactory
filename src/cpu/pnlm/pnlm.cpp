// SPDX-License-Identifier: GPL-2.0-only
// PNLM model version 1 (include/nss/params/pnlm.hpp). Pinned TU (no
// contraction): every fused operation is explicit (MulAdd, fmaf), in the
// order the model fixes, so that the CUDA kernels give the same bits.
//
// The references are taken a grid row at a time. A row's estimates go to a
// ring of kRing rows; the output rows that no later reference covers are
// gathered as soon as the row is done.
#include "nss/cpu_pnlm.hpp"
#include "nss/params/pnlm.hpp"
#include "cpu/hwy_config.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/pnlm/pnlm.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

#ifndef NSS_PNLM_ONCE
#define NSS_PNLM_ONCE
namespace nss {
namespace pnlm_detail {

constexpr int kRing = 4;  // grid rows whose estimates are kept: every output row is covered by at most 4
constexpr int kPad = 32;  // floats after a buffer, for whole vectors (and their neighbours' reads)

inline std::size_t align16(std::size_t n) { return (n + 15) & ~std::size_t{15}; }

// The work buffer, in floats: the estimate ring, the column sums, the keys
// kept (two floats each, kPnlmMaxGroup - 1 a reference) and their counts.
struct Layout {
    std::size_t ring, cols, best, kept, total;
};
inline Layout layout(int width, int block) {
    const std::size_t nx = static_cast<std::size_t>(pnlm_grid_count(width, block));
    Layout l{};
    l.ring = 0;
    l.cols = align16(static_cast<std::size_t>(kRing) * nx * block * block);
    l.best = l.cols + align16(5 * (static_cast<std::size_t>(width) + kPad));
    l.kept = l.best + align16(nx * (kPnlmMaxGroup - 1) * 2);
    l.total = l.kept + align16(nx) + 16;
    return l;
}

// Grid entries along an axis whose patch covers a position, in the gather
// order: lo .. hi, then the clamped last entry (last >= 0) when it is not
// among them. Entries that do not cover it are skipped.
struct Cover {
    int lo, hi, last;
};

}  // namespace pnlm_detail
}  // namespace nss
#endif

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

namespace {

using pnlm_detail::kPad;
using pnlm_detail::kRing;

template <int Q>
struct Plane {
    static constexpr int P = 2 * Q, A = P * P, L = pnlm_lanes(A);
    const PnlmImage& img;
    PnlmScale scale;
    int nx, ny;
    int K;              // group - 1
    float* ring;        // kRing x nx x A estimates
    float* cols;        // 5 x (width + kPad): one displacement's column sums of a grid row's two quarter
                        // rows, their quarter sums, and the patch totals
    std::uint64_t* best;  // nx x K: the keys kept for each reference of the grid row, sorted
    int* kept;          // nx

    float* est(int iy, int ix) const {
        return ring + (static_cast<std::size_t>(iy % kRing) * nx + ix) * A;
    }

    // The K best candidates of every reference of the grid row at y0. For each
    // frame and displacement, the column sums (fmaf down a quarter's rows) of
    // the row's two quarter rows are formed once for the whole width; each
    // reference then adds its four quarters (pnlm_columns, then (top left +
    // top right) + (bottom left + bottom right)).
    void match_row(int y0) {
        const hn::ScalableTag<float> df;
        const int N = static_cast<int>(hn::Lanes(df));
        const int W = img.width, H = img.height, a = img.a;
        const float* c0 = img.src[img.center];
        const std::ptrdiff_t cs = img.src_stride[img.center];
        const std::size_t row = static_cast<std::size_t>(W) + kPad;
        std::fill(kept, kept + nx, 0);
        for (int t = 0; t < img.frames; ++t) {
            const float* f = img.src[t];
            const std::ptrdiff_t fs = img.src_stride[t];
            for (int dy = -a; dy <= a; ++dy) {
                const int yc = y0 + dy;
                if (yc < 0 || yc > H - P) continue;
                for (int dx = -a; dx <= a; ++dx) {
                    // Columns x whose candidate column x + dx is inside the plane.
                    const int x_lo = std::max(0, -dx), x_hi = std::min(W, W - dx);
                    for (int qy = 0; qy < 2; ++qy) {
                        float* col = cols + qy * row;
                        const float* rrow = c0 + (y0 + qy * Q) * cs;
                        const float* crow = f + (yc + qy * Q) * fs + dx;
                        for (int x = x_lo; x < x_hi; x += N) {
                            const int n = std::min(N, x_hi - x);
                            auto sum = hn::Zero(df);
                            for (int r = 0; r < Q; ++r) {
                                const float* rp = rrow + r * cs + x;
                                const float* cp = crow + r * fs + x;
                                const auto rv = n == N ? hn::LoadU(df, rp) : hn::LoadN(df, rp, static_cast<std::size_t>(n));
                                const auto cv = n == N ? hn::LoadU(df, cp) : hn::LoadN(df, cp, static_cast<std::size_t>(n));
                                const auto d = hn::Sub(rv, cv);
                                sum = hn::MulAdd(d, d, sum);
                            }
                            hn::StoreU(sum, df, col + x);
                        }
                    }
                    const bool self = t == img.center && dx == 0 && dy == 0;
                    if (self) continue;
                    // Quarter sums (pnlm_columns) at every column, then the
                    // patch totals (top left + top right) + (bottom left +
                    // bottom right), a vector at a time; lanes past the valid
                    // range compute unused values inside the padding.
                    const int t_hi = x_hi - P + 1;  // totals for x < t_hi
                    for (int qy = 0; qy < 2; ++qy) {
                        const float* c = cols + qy * row;
                        float* q = cols + (2 + qy) * row;
                        for (int x = x_lo; x < t_hi + Q; x += N) {
                            if constexpr (Q == 4) {
                                hn::StoreU(hn::Add(hn::Add(hn::LoadU(df, c + x), hn::LoadU(df, c + x + 1)),
                                                   hn::Add(hn::LoadU(df, c + x + 2), hn::LoadU(df, c + x + 3))),
                                           df, q + x);
                            } else if constexpr (Q == 3) {
                                hn::StoreU(hn::Add(hn::Add(hn::LoadU(df, c + x), hn::LoadU(df, c + x + 1)),
                                                   hn::LoadU(df, c + x + 2)),
                                           df, q + x);
                            } else {
                                hn::StoreU(hn::Add(hn::LoadU(df, c + x), hn::LoadU(df, c + x + 1)), df, q + x);
                            }
                        }
                    }
                    const float* qt = cols + 2 * row;
                    const float* qb = cols + 3 * row;
                    float* tot = cols + 4 * row;
                    for (int x = x_lo; x < t_hi; x += N) {
                        hn::StoreU(hn::Add(hn::Add(hn::LoadU(df, qt + x), hn::LoadU(df, qt + x + Q)),
                                           hn::Add(hn::LoadU(df, qb + x), hn::LoadU(df, qb + x + Q))),
                                   df, tot + x);
                    }
                    // The references whose candidate x0 + dx is inside the plane: a run of the grid.
                    int ix0 = 0, ix1 = nx;
                    while (ix0 < nx && xs[ix0] + dx < 0) ++ix0;
                    while (ix1 > ix0 && xs[ix1 - 1] + dx > W - P) --ix1;
                    for (int ix = ix0; ix < ix1; ++ix) {
                        const int x0 = xs[ix], xc = x0 + dx;
                        const float total = tot[x0];
                        std::uint32_t bits;
                        std::memcpy(&bits, &total, sizeof(bits));
                        bits = std::min(bits, 0x7f800000u);
                        const std::uint64_t key = static_cast<std::uint64_t>(bits) << 32 | pnlm_pos(t, yc, xc);
                        std::uint64_t* b = best + static_cast<std::size_t>(ix) * K;
                        int& k = kept[ix];
                        if (k == K && !(key < b[K - 1])) continue;
                        int i = k < K ? k++ : K - 1;
                        for (; i > 0 && key < b[i - 1]; --i) b[i] = b[i - 1];
                        b[i] = key;
                    }
                }
            }
        }
    }

    // pnlm_weight of every match, a vector at a time (the same operations:
    // rint is round to nearest even, ldexp a product by 2^k, exact because
    // the result is a normal float).
    void weights(const std::uint64_t* best, int found, float* w) const {
        const hn::ScalableTag<float> df;
        const hn::RebindToSigned<decltype(df)> di;
        const int N = static_cast<int>(hn::Lanes(df));
        HWY_ALIGN float dist[kPnlmMaxGroup + kPad];
        for (int i = 0; i < found; ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(best[i] >> 32);
            std::memcpy(&dist[i], &bits, sizeof(float));
        }
        for (int i = found; i < kPnlmMaxGroup + kPad; ++i) dist[i] = 0.f;
        const auto inv_area = hn::Set(df, scale.inv_area), offset = hn::Set(df, scale.offset);
        const auto inv_h2 = hn::Set(df, scale.inv_h2), zero = hn::Zero(df);
        for (int i = 0; i < found; i += N) {
            const auto u = hn::Sub(hn::Mul(hn::Load(df, dist + i), inv_area), offset);
            const auto x = hn::Neg(hn::Mul(hn::IfThenElse(hn::Gt(u, zero), u, zero), inv_h2));
            const auto t = hn::Mul(x, hn::Set(df, 1.44269504088896341f));
            const auto k = hn::Round(t);
            const auto f = hn::Sub(t, k);
            auto q = hn::Set(df, 1.535336188319500e-4f);
            q = hn::MulAdd(q, f, hn::Set(df, 1.339887440266574e-3f));
            q = hn::MulAdd(q, f, hn::Set(df, 9.618437357674640e-3f));
            q = hn::MulAdd(q, f, hn::Set(df, 5.550332471162809e-2f));
            q = hn::MulAdd(q, f, hn::Set(df, 2.402264791363012e-1f));
            q = hn::MulAdd(q, f, hn::Set(df, 6.931472028550421e-1f));
            q = hn::MulAdd(q, f, hn::Set(df, 1.f));
            const auto ki = hn::ConvertTo(di, k);
            const auto scale2 = hn::BitCast(df, hn::ShiftLeft<23>(hn::Add(ki, hn::Set(di, 127))));
            const auto e = hn::Mul(q, scale2);
            // pnlm_exp is 0 at -87 and below.
            hn::Store(hn::IfThenElse(hn::Gt(x, hn::Set(df, -87.f)), e, zero), df, w + i);
        }
    }

    // The estimate of the reference patch at (x0, y0), whose found best
    // matches are in best, into out (A floats). The sums stay in registers:
    // acc per pixel, and the flat sums per lane (pnlm_lanes: pixel j in lane
    // j % L, the pixels L .. A - 1 second).
    void estimate(int x0, int y0, const std::uint64_t* best, int found, float* out) const {
        using D = hn::ScalableTag<float>;
        const D df;
        using V = hn::Vec<D>;
        constexpr int MN = static_cast<int>(hn::MaxLanes(D()));
        constexpr int NA = (A + MN - 1) / MN, NL = L / MN;
        constexpr int second = A > L ? A - L : 0;  // lanes with a second pixel
        constexpr int NS = (second + MN - 1) / MN;
        HWY_ALIGN float v[NA * MN + kPad];
        HWY_ALIGN float wts[kPnlmMaxGroup + kPad];
        std::memset(v, 0, sizeof(v));
        weights(best, found, wts);
        V acc[NA], s1[NL], s2[NL];
        for (int j = 0; j < NA; ++j) acc[j] = hn::Zero(df);
        for (int j = 0; j < NL; ++j) s1[j] = s2[j] = hn::Zero(df);
        float wsum = 0.f;
        for (int i = 0; i <= found; ++i) {
            int t = img.center, y = y0, x = x0;
            float w = 1.f;
            if (i > 0) {
                const std::uint32_t pos = static_cast<std::uint32_t>(best[i - 1]);
                t = pnlm_pos_t(pos);
                y = pnlm_pos_y(pos);
                x = pnlm_pos_x(pos);
                w = wts[i - 1];
            }
            const float* f = img.src[t] + y * img.src_stride[t] + x;
            const std::ptrdiff_t fs = img.src_stride[t];
            for (int r = 0; r < P; ++r) std::memcpy(v + r * P, f + r * fs, P * sizeof(float));
            wsum = wsum + w;
            const V vw = hn::Set(df, w);
            for (int j = 0; j < NA; ++j) acc[j] = hn::MulAdd(vw, hn::Load(df, v + j * MN), acc[j]);
            for (int j = 0; j < NL; ++j) {
                const V x1 = hn::Load(df, v + j * MN);
                s1[j] = hn::Add(s1[j], x1);
                s2[j] = hn::MulAdd(x1, x1, s2[j]);
            }
            for (int j = 0; j < NS; ++j) {
                const auto m = hn::FirstN(df, static_cast<std::size_t>(second - j * MN));
                const V x2 = hn::Load(df, v + L + j * MN);
                s1[j] = hn::IfThenElse(m, hn::Add(s1[j], x2), s1[j]);
                s2[j] = hn::IfThenElse(m, hn::MulAdd(x2, x2, s2[j]), s2[j]);
            }
        }
        // Butterfly over the L lanes: every lane ends with the same sum, that
        // of lane 0 (x + y and y + x are the same float), whose levels of a
        // vector or more are vector additions, the rest scalar.
        int nv = NL;
        while (nv > 1) {
            nv /= 2;
            for (int j = 0; j < nv; ++j) {
                s1[j] = hn::Add(s1[j], s1[j + nv]);
                s2[j] = hn::Add(s2[j], s2[j + nv]);
            }
        }
        HWY_ALIGN float l1[MN], l2[MN];
        hn::Store(s1[0], df, l1);
        hn::Store(s2[0], df, l2);
        for (int off = std::min(MN, L) / 2; off > 0; off >>= 1) {
            for (int l = 0; l < off; ++l) {
                l1[l] = l1[l] + l1[l + off];
                l2[l] = l2[l] + l2[l + off];
            }
        }
        const PnlmFlat flat = pnlm_flat(l1[0], l2[0], static_cast<float>((found + 1) * A), scale);
        // raw = acc / wsum, then alpha mean + (1 - alpha) raw (pnlm_mix).
        const V vs = hn::Set(df, wsum), va = hn::Set(df, flat.alpha_mean), vb = hn::Set(df, flat.one_minus);
        HWY_ALIGN float o[NA * MN];
        for (int j = 0; j < NA; ++j) hn::Store(hn::Add(va, hn::Mul(vb, hn::Div(acc[j], vs))), df, o + j * MN);
        std::memcpy(out, o, A * sizeof(float));
    }

    // Grid entries covering v along an axis of n pixels with count entries.
    static pnlm_detail::Cover cover(int v, int n, int count) {
        pnlm_detail::Cover c;
        c.lo = v - P + 1 <= 0 ? 0 : (v - P + 1 + Q - 1) / Q;
        c.hi = std::min(count - 1, v / Q);
        const int last = n - P;
        c.last = (c.hi < count - 1 && last <= v && v < last + P) ? count - 1 : -1;
        return c;
    }

    // Per column: the grid columns covering it; per position in a patch: its
    // tent weight; per grid column: its x.
    std::vector<pnlm_detail::Cover> covers_x;
    float tent[P];
    std::vector<int> xs;

    void prepare() {
        // Lanes past the valid columns read the padding: defined values.
        std::fill(cols, cols + 5 * (static_cast<std::size_t>(img.width) + kPad), 0.f);
        xs.resize(static_cast<std::size_t>(nx));
        for (int ix = 0; ix < nx; ++ix) xs[static_cast<std::size_t>(ix)] = pnlm_grid_pos(ix, img.width, P);
        covers_x.resize(static_cast<std::size_t>(img.width));
        for (int x = 0; x < img.width; ++x) covers_x[static_cast<std::size_t>(x)] = cover(x, img.width, nx);
        for (int i = 0; i < P; ++i) tent[i] = pnlm_tent(i, P);
    }

    // Output row y from the estimates in the ring.
    void gather(int y) const {
        const int W = img.width, H = img.height;
        const pnlm_detail::Cover cy = cover(y, H, ny);
        float* out = img.out + y * img.out_stride;
        for (int x = 0; x < W; ++x) {
            const pnlm_detail::Cover& cx = covers_x[static_cast<std::size_t>(x)];
            float n = 0.f, d = 0.f;
            for (int ia = 0; ia <= cy.hi - cy.lo + (cy.last >= 0); ++ia) {
                const int iy = cy.lo + ia <= cy.hi ? cy.lo + ia : cy.last;
                const int gy = pnlm_grid_pos(iy, H, P);
                if (gy > y || y >= gy + P) continue;
                const float wy = tent[y - gy];
                for (int ib = 0; ib <= cx.hi - cx.lo + (cx.last >= 0); ++ib) {
                    const int ix = cx.lo + ib <= cx.hi ? cx.lo + ib : cx.last;
                    const int gx = pnlm_grid_pos(ix, W, P);
                    if (gx > x || x >= gx + P) continue;
                    const float wxy = wy * tent[x - gx];
                    d = d + wxy;
                    n = fmaf(wxy, est(iy, ix)[(y - gy) * P + (x - gx)], n);
                }
            }
            out[x] = n / d;
        }
    }

    void run() {
        prepare();
        const int H = img.height;
        int next_row = 0;  // the first output row not gathered yet
        for (int iy = 0; iy < ny; ++iy) {
            const int y0 = pnlm_grid_pos(iy, H, P);
            match_row(y0);
            for (int ix = 0; ix < nx; ++ix)
                estimate(xs[static_cast<std::size_t>(ix)], y0, best + static_cast<std::size_t>(ix) * K, kept[ix], est(iy, ix));
            // Rows above the next grid row's top are complete.
            const int end = iy + 1 < ny ? pnlm_grid_pos(iy + 1, H, P) : H;
            for (; next_row < end; ++next_row) gather(next_row);
        }
    }
};

template <int Q>
void run_plane(const PnlmImage& img, float* work) {
    const pnlm_detail::Layout l = pnlm_detail::layout(img.width, 2 * Q);
    Plane<Q> p{img,
               pnlm_scale(img.sigma, 2 * Q),
               pnlm_grid_count(img.width, 2 * Q),
               pnlm_grid_count(img.height, 2 * Q),
               img.group - 1,
               work + l.ring,
               work + l.cols,
               reinterpret_cast<std::uint64_t*>(work + l.best),
               reinterpret_cast<int*>(work + l.kept),
               {},
               {},
               {}};
    p.run();
}

}  // namespace

void PnlmDenoise(const PnlmImage& img, float* work, std::size_t work_floats) {
    (void)work_floats;
    switch (img.block) {
    case 4: run_plane<2>(img, work); break;
    case 6: run_plane<3>(img, work); break;
    case 8: run_plane<4>(img, work); break;
    default: throw std::invalid_argument("nss.PNLM: block_size must be 4, 6 or 8");
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(PnlmDenoise);

std::size_t pnlm_work_floats(int width, int block) { return pnlm_detail::layout(width, block).total; }

void pnlm_denoise(const PnlmImage& img, float* work, std::size_t work_floats) {
    if (img.width < img.block || img.height < img.block || img.width > kPnlmMaxSide || img.height > kPnlmMaxSide ||
        img.frames < 1 || img.frames > 2 * kPnlmMaxD + 1 || img.center < 0 || img.center >= img.frames ||
        img.group < 2 || img.group > kPnlmMaxGroup || img.a < 1 || !(img.sigma > 0.f) ||
        work_floats < pnlm_work_floats(img.width, img.block)) {
        throw std::invalid_argument("nss.PNLM: invalid plane geometry or arguments");
    }
    HWY_DYNAMIC_DISPATCH(PnlmDenoise)(img, work, work_floats);
}
}  // namespace nss
#endif
