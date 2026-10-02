// SPDX-License-Identifier: GPL-2.0-only
#include "cpu/wnnm/jacobi8.hpp"
#include "cpu/wnnm/numerics.hpp"
#include "cpu/hwy_config.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>

#ifndef NSS_SVD_QREPLAY_ROW_MAJOR
#define NSS_SVD_QREPLAY_ROW_MAJOR 0
#endif

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/wnnm/svd_batch8.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

namespace {

#if HWY_TARGET == HWY_NEON || HWY_TARGET == HWY_NEON_WITHOUT_AES
constexpr int kBatchLanes = 4;
#else
constexpr int kBatchLanes = 16;
#endif
constexpr int kN = 8;

inline std::size_t TallIndex(int row, int col, int lane) {
    return (static_cast<std::size_t>(col) * kSvdBatch8MaxM + row) * kBatchLanes + lane;
}

inline std::size_t SmallIndex(int row, int col, int lane) {
    return (static_cast<std::size_t>(col) * kN + row) * kBatchLanes + lane;
}

#if HWY_MAX_BYTES >= 16
// In-register 4x4 transpose (rows a,b,c,e become columns). Pure data
// movement, so the SoA pack/unpack below is bit-identical to the scalar copy.
using D4 = hn::FixedTag<float, 4>;
HWY_INLINE void Transpose4x4(hn::Vec<D4>& a, hn::Vec<D4>& b, hn::Vec<D4>& c, hn::Vec<D4>& e) {
    const D4 d;
    const hn::Repartition<std::uint64_t, D4> d64;
    const auto ab0 = hn::BitCast(d64, hn::InterleaveLower(d, a, b));
    const auto ab1 = hn::BitCast(d64, hn::InterleaveUpper(d, a, b));
    const auto ce0 = hn::BitCast(d64, hn::InterleaveLower(d, c, e));
    const auto ce1 = hn::BitCast(d64, hn::InterleaveUpper(d, c, e));
    a = hn::BitCast(d, hn::InterleaveLower(d64, ab0, ce0));
    b = hn::BitCast(d, hn::InterleaveUpper(d64, ab0, ce0));
    c = hn::BitCast(d, hn::InterleaveLower(d64, ab1, ce1));
    e = hn::BitCast(d, hn::InterleaveUpper(d64, ab1, ce1));
}
#endif

template <bool kNeedVt, class D, bool kNeedU = true>
bool SvdChunk(D d, int m, const float* const* A, const int* lda, float* const* U, const int* ldu,
              float* const* S, float* const* Vt, const int* ldvt, int count) {
    using V = hn::Vec<D>;
    using M = hn::Mask<D>;

    alignas(64) float tall[kSvdBatch8MaxM * kN * kBatchLanes];
    alignas(64) float reflectors[kSvdBatch8MaxM * kN * kBatchLanes];
    alignas(64) float rsoa[kN * kN * kBatchLanes];
    alignas(64) float vsoa[kN * kN * kBatchLanes];
    alignas(64) float ssoa[kN * kBatchLanes];
    alignas(64) float beta[kN * kBatchLanes];
    // Full 64x8 chunks only read elements written by pack + QR. Partial chunks
    // initialize the inactive rows/lanes as finite sentinels. The private
    // storage width is four matrices on NEON and sixteen on the wider targets.
    const bool packed_hot = (count == kBatchLanes && m == kSvdBatch8MaxM);
    if (!packed_hot) {
        std::memset(tall, 0, sizeof(tall));
        std::memset(reflectors, 0, sizeof(reflectors));
    }
    std::memset(rsoa, 0, sizeof(rsoa));
    std::memset(ssoa, 0, sizeof(ssoa));
    std::memset(beta, 0, sizeof(beta));

    float scales[kBatchLanes];
#if HWY_MAX_BYTES >= 16
    if (packed_hot) {
        // Full chunk: transpose 4 lanes x 4 rows at a time into the SoA tile
        // and keep each lane's magnitude maximum in the matching vector lane.
        const D4 d4;
        const hn::RebindToUnsigned<D4> du4;
        const auto magnitude_mask = hn::Set(du4, 0x7fffffffu);
        for (int lane0 = 0; lane0 < kBatchLanes; lane0 += 4) {
            auto vmax = hn::Zero(du4);
            for (int col = 0; col < kN; ++col) {
                const float* a0 = A[lane0] + col * lda[lane0];
                const float* a1 = A[lane0 + 1] + col * lda[lane0 + 1];
                const float* a2 = A[lane0 + 2] + col * lda[lane0 + 2];
                const float* a3 = A[lane0 + 3] + col * lda[lane0 + 3];
                for (int row = 0; row < kSvdBatch8MaxM; row += 4) {
                    auto r0 = hn::LoadU(d4, a0 + row);
                    auto r1 = hn::LoadU(d4, a1 + row);
                    auto r2 = hn::LoadU(d4, a2 + row);
                    auto r3 = hn::LoadU(d4, a3 + row);
                    Transpose4x4(r0, r1, r2, r3);
                    vmax = hn::Max(vmax, hn::And(hn::BitCast(du4, r0), magnitude_mask));
                    vmax = hn::Max(vmax, hn::And(hn::BitCast(du4, r1), magnitude_mask));
                    vmax = hn::Max(vmax, hn::And(hn::BitCast(du4, r2), magnitude_mask));
                    vmax = hn::Max(vmax, hn::And(hn::BitCast(du4, r3), magnitude_mask));
                    hn::StoreU(r0, d4, tall + TallIndex(row, col, lane0));
                    hn::StoreU(r1, d4, tall + TallIndex(row + 1, col, lane0));
                    hn::StoreU(r2, d4, tall + TallIndex(row + 2, col, lane0));
                    hn::StoreU(r3, d4, tall + TallIndex(row + 3, col, lane0));
                }
            }
            alignas(16) std::uint32_t maxima[4];
            hn::Store(vmax, du4, maxima);
            for (int i = 0; i < 4; ++i) {
                const int lane = lane0 + i;
                if (maxima[i] >= 0x7f800000u) return false;
                scales[lane] = detail::svd_input_scale(maxima[i]);
                if (scales[lane] != 1.f) {
                    for (int col = 0; col < kN; ++col) {
                        for (int row = 0; row < m; ++row) tall[TallIndex(row, col, lane)] *= scales[lane];
                    }
                }
            }
        }
    } else
#endif
    for (int lane = 0; lane < count; ++lane) {
        std::uint32_t maximum = 0;
        for (int col = 0; col < kN; ++col) {
            for (int row = 0; row < m; ++row) {
                const float value = A[lane][row + col * lda[lane]];
                maximum = std::max(maximum, detail::svd_magnitude_bits(value));
                tall[TallIndex(row, col, lane)] = value;
            }
        }
        if (maximum >= 0x7f800000u) return false;
        scales[lane] = detail::svd_input_scale(maximum);
        if (scales[lane] != 1.f) {
            for (int col = 0; col < kN; ++col) {
                for (int row = 0; row < m; ++row) tall[TallIndex(row, col, lane)] *= scales[lane];
            }
        }
    }

    const M valid = hn::FirstN(d, static_cast<std::size_t>(count));
    const V zero = hn::Zero(d);
    const V one = hn::Set(d, 1.0f);
    const V minus_one = hn::Set(d, -1.0f);

    // Householder QR. Each vector lane owns one independent matrix, so the
    // reduction order within a matrix stays scalar and deterministic.
    for (int k = 0; k < kN; ++k) {
        V norm2 = zero;
        for (int row = k; row < m; ++row) {
            const V x = hn::Load(d, tall + TallIndex(row, k, 0));
            norm2 = hn::MulAdd(x, x, norm2);
        }
        const V norm = hn::Sqrt(hn::Max(norm2, zero));
        const V x0 = hn::Load(d, tall + TallIndex(k, k, 0));
        const V sign = hn::IfThenElse(hn::Ge(x0, zero), one, minus_one);
        M usable = hn::And(valid, hn::Ge(norm, hn::Set(d, 1e-20f)));

        for (int row = k; row < m; ++row) {
            V value = hn::Load(d, tall + TallIndex(row, k, 0));
            if (row == k) {
                value = hn::Add(value, hn::Mul(sign, norm));
            }
            hn::Store(value, d, reflectors + TallIndex(row, k, 0));
        }

        V vtv = zero;
        for (int row = k; row < m; ++row) {
            const V value = hn::Load(d, reflectors + TallIndex(row, k, 0));
            vtv = hn::MulAdd(value, value, vtv);
        }
        usable = hn::And(usable, hn::Ge(vtv, hn::Set(d, 1e-30f)));
        const V safe_vtv = hn::IfThenElse(usable, vtv, one);
        const V b = hn::IfThenElse(usable, hn::Div(hn::Set(d, 2.0f), safe_vtv), zero);
        hn::Store(b, d, beta + static_cast<std::size_t>(k) * kBatchLanes);

        // Row-major dot/update: each column keeps its own row-ascending FMA
        // chain (bit-identical values), and interleaving the independent
        // chains fills FMA latency instead of serializing per column.
        V dots[kN];
        for (int col = k; col < kN; ++col) {
            dots[col] = zero;
        }
        for (int row = k; row < m; ++row) {
            const V reflector = hn::Load(d, reflectors + TallIndex(row, k, 0));
            for (int col = k; col < kN; ++col) {
                dots[col] = hn::MulAdd(reflector, hn::Load(d, tall + TallIndex(row, col, 0)), dots[col]);
            }
        }
        for (int col = k; col < kN; ++col) {
            dots[col] = hn::Mul(b, dots[col]);
        }
        for (int row = k; row < m; ++row) {
            const V reflector = hn::Load(d, reflectors + TallIndex(row, k, 0));
            for (int col = k; col < kN; ++col) {
                const V value = hn::Load(d, tall + TallIndex(row, col, 0));
                hn::Store(hn::NegMulAdd(dots[col], reflector, value), d, tall + TallIndex(row, col, 0));
            }
        }

        hn::Store(hn::IfThenElse(usable, hn::Neg(hn::Mul(sign, norm)), zero), d,
                  rsoa + SmallIndex(k, k, 0));
        for (int col = k + 1; col < kN; ++col) {
            hn::Store(hn::Load(d, tall + TallIndex(k, col, 0)), d, rsoa + SmallIndex(k, col, 0));
        }
    }

    V ju[kN * kN];
    V jv[kN * kN];
    V norms[kN];
    for (int col = 0; col < kN; ++col) {
        norms[col] = zero;
        for (int row = 0; row < kN; ++row) {
            ju[col * kN + row] = hn::Load(d, rsoa + SmallIndex(row, col, 0));
            if constexpr (kNeedVt) {
                jv[col * kN + row] = (row == col) ? one : zero;
            }
            norms[col] = hn::MulAdd(ju[col * kN + row], ju[col * kN + row], norms[col]);
        }
    }

    V max_norm = zero;
    for (int col = 0; col < kN; ++col) max_norm = hn::Max(max_norm, norms[col]);
    const V rank_floor = hn::Mul(max_norm, hn::Set(d, detail::kSvdRankFloorSquared));

    static constexpr int pairs[7][4][2] = {
        {{0, 1}, {2, 3}, {4, 5}, {6, 7}}, {{0, 2}, {1, 3}, {4, 6}, {5, 7}},
        {{0, 3}, {1, 2}, {4, 7}, {5, 6}}, {{0, 4}, {1, 5}, {2, 6}, {3, 7}},
        {{0, 5}, {1, 4}, {2, 7}, {3, 6}}, {{0, 6}, {1, 7}, {2, 4}, {3, 5}},
        {{0, 7}, {1, 6}, {2, 5}, {3, 4}},
    };
    M active = valid;
    for (int sweep = 0; sweep < 32; ++sweep) {
        M rotated = hn::FirstN(d, 0);
        for (int round = 0; round < 7; ++round) {
            for (int pair = 0; pair < 4; ++pair) {
                const int p = pairs[round][pair][0];
                const int q = pairs[round][pair][1];
                const V app = norms[p];
                const V aqq = norms[q];
                V apq = zero;
                for (int row = 0; row < kN; ++row) {
                    apq = hn::MulAdd(ju[p * kN + row], ju[q * kN + row], apq);
                }
                const V threshold = hn::Mul(hn::Set(d, detail::kJacobiCorrelationTolerance),
                                            hn::Sqrt(hn::Max(hn::Mul(app, aqq), zero)));
                const M rotate = hn::And(hn::And(active, hn::Gt(hn::Abs(apq), threshold)),
                                         hn::And(hn::Gt(app, rank_floor), hn::Gt(aqq, rank_floor)));
                rotated = hn::Or(rotated, rotate);
                const V delta = hn::Mul(hn::Sub(aqq, app), hn::Set(d, 0.5f));
                const V angle_scale = hn::IfThenElse(rotate, hn::Max(hn::Abs(delta), hn::Abs(apq)), one);
                const V x = hn::Div(delta, angle_scale);
                const V y = hn::Div(hn::IfThenElse(rotate, apq, one), angle_scale);
                const V t = hn::Div(y, hn::Add(x, hn::CopySign(hn::Sqrt(hn::MulAdd(x, x, hn::Mul(y, y))), x)));
                const V cs = hn::Div(one, hn::Sqrt(hn::MulAdd(t, t, one)));
                const V sn = hn::Mul(cs, t);
                for (int row = 0; row < kN; ++row) {
                    const V up = ju[p * kN + row];
                    const V uq = ju[q * kN + row];
                    ju[p * kN + row] = hn::IfThenElse(rotate, hn::NegMulAdd(sn, uq, hn::Mul(cs, up)), up);
                    ju[q * kN + row] = hn::IfThenElse(rotate, hn::MulAdd(sn, up, hn::Mul(cs, uq)), uq);
                    if constexpr (kNeedVt) {
                        const V vp = jv[p * kN + row];
                        const V vq = jv[q * kN + row];
                        jv[p * kN + row] = hn::IfThenElse(rotate, hn::NegMulAdd(sn, vq, hn::Mul(cs, vp)), vp);
                        jv[q * kN + row] = hn::IfThenElse(rotate, hn::MulAdd(sn, vp, hn::Mul(cs, vq)), vq);
                    }
                }
                // The rotation is orthogonal, so update the two squared
                // column norms analytically. A complete recomputation after
                // every pairing round dominated the 8-column Jacobi path;
                // the sweep-end recomputation below retains the rank-deficient
                // guard while removing six redundant dot-product sets.
                const V c2 = hn::Mul(cs, cs);
                const V s2 = hn::Mul(sn, sn);
                const V cs2 = hn::Mul(hn::Set(d, 2.0f), hn::Mul(cs, sn));
                const V np = hn::Max(hn::Add(hn::Sub(hn::Mul(c2, app), hn::Mul(cs2, apq)),
                                              hn::Mul(s2, aqq)), zero);
                const V nq = hn::Max(hn::Add(hn::Add(hn::Mul(s2, app), hn::Mul(cs2, apq)),
                                              hn::Mul(c2, aqq)), zero);
                norms[p] = hn::IfThenElse(rotate, np, norms[p]);
                norms[q] = hn::IfThenElse(rotate, nq, norms[q]);
            }
        }
        // Refresh once per sweep to absorb roundoff and preserve the
        // rank-floor behavior for nearly deficient groups.
        for (int col = 0; col < kN; ++col) {
            norms[col] = zero;
            for (int row = 0; row < kN; ++row) {
                norms[col] = hn::MulAdd(ju[col * kN + row], ju[col * kN + row], norms[col]);
            }
        }
        active = hn::And(active, rotated);
        if (hn::AllFalse(d, active)) {
            break;
        }
    }

    for (int col = 0; col < kN; ++col) {
        const V singular = hn::IfThenElse(hn::Gt(norms[col], rank_floor),
                                             hn::Sqrt(hn::Max(norms[col], zero)), zero);
        const M nonzero = hn::And(valid, hn::Gt(singular, hn::Set(d, 1e-20f)));
        const V inv = hn::IfThenElse(nonzero, hn::Div(one, hn::IfThenElse(nonzero, singular, one)), one);
        hn::Store(singular, d, ssoa + static_cast<std::size_t>(col) * kBatchLanes);
        for (int row = 0; row < kN; ++row) {
            hn::Store(hn::IfThenElse(nonzero, hn::Mul(ju[col * kN + row], inv), zero), d,
                      rsoa + SmallIndex(row, col, 0));
            if constexpr (kNeedVt) {
                hn::Store(jv[col * kN + row], d, vsoa + SmallIndex(row, col, 0));
            }
        }
    }

    // Sorting is lane-dependent. It is only 8 columns, so perform the stable
    // selection step on the packed scalar lanes before replaying Q.
    for (int lane = 0; lane < count; ++lane) {
        for (int a = 0; a < kN; ++a) {
            int best = a;
            for (int bcol = a + 1; bcol < kN; ++bcol) {
                if (ssoa[static_cast<std::size_t>(bcol) * kBatchLanes + lane] >
                    ssoa[static_cast<std::size_t>(best) * kBatchLanes + lane]) {
                    best = bcol;
                }
            }
            if (best != a) {
                std::swap(ssoa[static_cast<std::size_t>(a) * kBatchLanes + lane],
                          ssoa[static_cast<std::size_t>(best) * kBatchLanes + lane]);
                for (int row = 0; row < kN; ++row) {
                    std::swap(rsoa[SmallIndex(row, a, lane)], rsoa[SmallIndex(row, best, lane)]);
                    if constexpr (kNeedVt) {
                        std::swap(vsoa[SmallIndex(row, a, lane)], vsoa[SmallIndex(row, best, lane)]);
                    }
                }
            }
        }
    }

    if constexpr (!kNeedU) {
        // Singular values and right vectors only: no Q replay, no U output.
        for (int lane = 0; lane < count; ++lane) {
            for (int col = 0; col < kN; ++col) {
                S[lane][col] = ssoa[static_cast<std::size_t>(col) * kBatchLanes + lane] / scales[lane];
                if (detail::svd_magnitude_bits(S[lane][col]) >= 0x7f800000u) return false;
            }
            for (int col = 0; col < kN; ++col) {
                for (int row = 0; row < kN; ++row) {
                    Vt[lane][row + col * ldvt[lane]] = vsoa[SmallIndex(col, row, lane)];
                }
            }
        }
        return true;
    }
    std::memset(tall, 0, sizeof(tall));
    for (int col = 0; col < kN; ++col) {
        for (int row = 0; row < kN; ++row) {
            hn::Store(hn::Load(d, rsoa + SmallIndex(row, col, 0)), d, tall + TallIndex(row, col, 0));
        }
    }
    for (int k = kN - 1; k >= 0; --k) {
        const V b = hn::Load(d, beta + static_cast<std::size_t>(k) * kBatchLanes);
#if NSS_SVD_QREPLAY_ROW_MAJOR
        V factors[kN];
        for (int col = 0; col < kN; ++col) {
            factors[col] = zero;
        }
        for (int row = k; row < m; ++row) {
            const V reflector = hn::Load(d, reflectors + TallIndex(row, k, 0));
            for (int col = 0; col < kN; ++col) {
                factors[col] = hn::MulAdd(reflector, hn::Load(d, tall + TallIndex(row, col, 0)), factors[col]);
            }
        }
        for (int col = 0; col < kN; ++col) {
            factors[col] = hn::Mul(b, factors[col]);
        }
        for (int row = k; row < m; ++row) {
            const V reflector = hn::Load(d, reflectors + TallIndex(row, k, 0));
            for (int col = 0; col < kN; ++col) {
                const V value = hn::Load(d, tall + TallIndex(row, col, 0));
                hn::Store(hn::NegMulAdd(factors[col], reflector, value), d, tall + TallIndex(row, col, 0));
            }
        }
#else
        for (int col = 0; col < kN; ++col) {
            V dot = zero;
            for (int row = k; row < m; ++row) {
                dot = hn::MulAdd(hn::Load(d, reflectors + TallIndex(row, k, 0)),
                                 hn::Load(d, tall + TallIndex(row, col, 0)), dot);
            }
            const V factor = hn::Mul(b, dot);
            for (int row = k; row < m; ++row) {
                const V value = hn::Load(d, tall + TallIndex(row, col, 0));
                hn::Store(hn::NegMulAdd(factor, hn::Load(d, reflectors + TallIndex(row, k, 0)), value), d,
                          tall + TallIndex(row, col, 0));
            }
        }
#endif
    }

#if HWY_MAX_BYTES >= 16
    if (count == kBatchLanes && (m & 3) == 0) {
        // A false return makes the caller recompute every item, so writing
        // all S before U changes no observable result.
        for (int lane = 0; lane < count; ++lane) {
            for (int col = 0; col < kN; ++col) {
                S[lane][col] = ssoa[static_cast<std::size_t>(col) * kBatchLanes + lane] / scales[lane];
                if (detail::svd_magnitude_bits(S[lane][col]) >= 0x7f800000u) return false;
            }
        }
        const D4 d4;
        for (int lane0 = 0; lane0 < kBatchLanes; lane0 += 4) {
            for (int col = 0; col < kN; ++col) {
                float* u0 = U[lane0] + col * ldu[lane0];
                float* u1 = U[lane0 + 1] + col * ldu[lane0 + 1];
                float* u2 = U[lane0 + 2] + col * ldu[lane0 + 2];
                float* u3 = U[lane0 + 3] + col * ldu[lane0 + 3];
                for (int row = 0; row < m; row += 4) {
                    auto r0 = hn::LoadU(d4, tall + TallIndex(row, col, lane0));
                    auto r1 = hn::LoadU(d4, tall + TallIndex(row + 1, col, lane0));
                    auto r2 = hn::LoadU(d4, tall + TallIndex(row + 2, col, lane0));
                    auto r3 = hn::LoadU(d4, tall + TallIndex(row + 3, col, lane0));
                    Transpose4x4(r0, r1, r2, r3);
                    hn::StoreU(r0, d4, u0 + row);
                    hn::StoreU(r1, d4, u1 + row);
                    hn::StoreU(r2, d4, u2 + row);
                    hn::StoreU(r3, d4, u3 + row);
                }
            }
        }
        if constexpr (kNeedVt) {
            for (int lane = 0; lane < count; ++lane) {
                for (int col = 0; col < kN; ++col) {
                    for (int row = 0; row < kN; ++row) {
                        Vt[lane][row + col * ldvt[lane]] = vsoa[SmallIndex(col, row, lane)];
                    }
                }
            }
        }
        return true;
    }
#endif
    for (int lane = 0; lane < count; ++lane) {
        for (int col = 0; col < kN; ++col) {
            S[lane][col] = ssoa[static_cast<std::size_t>(col) * kBatchLanes + lane] / scales[lane];
            if (detail::svd_magnitude_bits(S[lane][col]) >= 0x7f800000u) return false;
            for (int row = 0; row < m; ++row) {
                U[lane][row + col * ldu[lane]] = tall[TallIndex(row, col, lane)];
            }
        }
        if constexpr (kNeedVt) {
            for (int col = 0; col < kN; ++col) {
                for (int row = 0; row < kN; ++row) {
                    Vt[lane][row + col * ldvt[lane]] = vsoa[SmallIndex(col, row, lane)];
                }
            }
        }
    }
    return true;
}

}  // namespace

int SvdEconomy8Batch(int m, const float* const* A, const int* lda, float* const* U, const int* ldu, float* const* S,
                     float* const* Vt, const int* ldvt, int count) {
    if (m < kN || m > kSvdBatch8MaxM || !A || !lda || !U || !ldu || !S || !Vt || !ldvt || count < 1) {
        return -1;
    }
    const hn::CappedTag<float, kBatchLanes> d;
    const int lanes = static_cast<int>(hn::Lanes(d));
    for (int begin = 0; begin < count; begin += lanes) {
        const int chunk = std::min(lanes, count - begin);
        if (!SvdChunk<true>(d, m, A + begin, lda + begin, U + begin, ldu + begin, S + begin, Vt + begin, ldvt + begin,
                            chunk)) return -1;
    }
    return 0;
}

int SvdEconomy8BatchU(int m, const float* const* A, const int* lda, float* const* U, const int* ldu, float* const* S,
                      int count) {
    if (m < kN || m > kSvdBatch8MaxM || !A || !lda || !U || !ldu || !S || count < 1) {
        return -1;
    }
    const hn::CappedTag<float, kBatchLanes> d;
    const int lanes = static_cast<int>(hn::Lanes(d));
    for (int begin = 0; begin < count; begin += lanes) {
        const int chunk = std::min(lanes, count - begin);
        if (!SvdChunk<false>(d, m, A + begin, lda + begin, U + begin, ldu + begin, S + begin, nullptr, nullptr, chunk)) return -1;
    }
    return 0;
}

#if !NSS_SVD_QREPLAY_ROW_MAJOR
// Singular values plus V^T only (WNNM reconstructs from A*V, see batch.cpp).
int SvdEconomy8BatchSV(int m, const float* const* A, const int* lda, float* const* S, float* const* Vt,
                       const int* ldvt, int count) {
    if (m < kN || m > kSvdBatch8MaxM || !A || !lda || !S || !Vt || !ldvt || count < 1) {
        return -1;
    }
    const hn::CappedTag<float, kBatchLanes> d;
    const int lanes = static_cast<int>(hn::Lanes(d));
    for (int begin = 0; begin < count; begin += lanes) {
        const int chunk = std::min(lanes, count - begin);
        if (!SvdChunk<true, decltype(d), false>(d, m, A + begin, lda + begin, nullptr, nullptr, S + begin,
                                                 Vt + begin, ldvt + begin, chunk)) return -1;
    }
    return 0;
}
#endif

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
#if NSS_SVD_QREPLAY_ROW_MAJOR
namespace {
using SvdBatchFunction = int (*)(int, const float* const*, const int*, float* const*, const int*,
                                 float* const*, float* const*, const int*, int);
using SvdBatchUFunction =
    int (*)(int, const float* const*, const int*, float* const*, const int*, float* const*, int);

std::atomic<SvdBatchFunction> qreplay_dispatch[HWY_MAX_DYNAMIC_TARGETS + 2]{};
std::atomic<SvdBatchUFunction> qreplay_u_dispatch[HWY_MAX_DYNAMIC_TARGETS + 2]{};

#define NSS_SVD_QREPLAY_SELECT(TARGET, NAMESPACE)                                                          \
    if (target == (TARGET)) {                                                                              \
        return &NAMESPACE::SvdEconomy8Batch;                                                               \
    }
#define NSS_SVD_QREPLAY_SELECT_U(TARGET, NAMESPACE)                                                        \
    if (target == (TARGET)) {                                                                              \
        return &NAMESPACE::SvdEconomy8BatchU;                                                              \
    }

SvdBatchFunction SelectQReplay(std::size_t& index) {
    const auto supported = HWY_SUPPORTED_TARGETS;
    hwy::GetChosenTarget().Update(supported);
    index = hwy::GetChosenTarget().GetIndex();
    const auto generated = supported & HWY_TARGETS;
    const auto target = generated & -generated;
    HWY_VISIT_TARGETS(NSS_SVD_QREPLAY_SELECT)
    return nullptr;
}

SvdBatchUFunction SelectQReplayU(std::size_t& index) {
    const auto supported = HWY_SUPPORTED_TARGETS;
    hwy::GetChosenTarget().Update(supported);
    index = hwy::GetChosenTarget().GetIndex();
    const auto generated = supported & HWY_TARGETS;
    const auto target = generated & -generated;
    HWY_VISIT_TARGETS(NSS_SVD_QREPLAY_SELECT_U)
    return nullptr;
}

#undef NSS_SVD_QREPLAY_SELECT_U
#undef NSS_SVD_QREPLAY_SELECT
}  // namespace

int svd_economy_8_batch_hwy(int m, const float* const* A, const int* lda, float* const* U, const int* ldu,
                            float* const* S, float* const* Vt, const int* ldvt, int count) {
    std::size_t index = hwy::GetChosenTarget().GetIndex();
    auto dispatch = qreplay_dispatch[index].load(std::memory_order_relaxed);
    if (!dispatch) {
        dispatch = SelectQReplay(index);
        if (!dispatch) {
            return -1;
        }
        qreplay_dispatch[index].store(dispatch, std::memory_order_relaxed);
    }
    return dispatch(m, A, lda, U, ldu, S, Vt, ldvt, count);
}

int svd_economy_8_batch_u_hwy(int m, const float* const* A, const int* lda, float* const* U, const int* ldu,
                              float* const* S, int count) {
    std::size_t index = hwy::GetChosenTarget().GetIndex();
    auto dispatch = qreplay_u_dispatch[index].load(std::memory_order_relaxed);
    if (!dispatch) {
        dispatch = SelectQReplayU(index);
        if (!dispatch) {
            return -1;
        }
        qreplay_u_dispatch[index].store(dispatch, std::memory_order_relaxed);
    }
    return dispatch(m, A, lda, U, ldu, S, count);
}
#else
HWY_EXPORT(SvdEconomy8Batch);
HWY_EXPORT(SvdEconomy8BatchU);
HWY_EXPORT(SvdEconomy8BatchSV);

int svd_economy_8_batch_sv_hwy(int m, const float* const* A, const int* lda, float* const* S, float* const* Vt,
                               const int* ldvt, int count) {
    return HWY_DYNAMIC_DISPATCH(SvdEconomy8BatchSV)(m, A, lda, S, Vt, ldvt, count);
}

int svd_economy_8_batch_hwy(int m, const float* const* A, const int* lda, float* const* U, const int* ldu,
                            float* const* S, float* const* Vt, const int* ldvt, int count) {
    return HWY_DYNAMIC_DISPATCH(SvdEconomy8Batch)(m, A, lda, U, ldu, S, Vt, ldvt, count);
}

int svd_economy_8_batch_u_hwy(int m, const float* const* A, const int* lda, float* const* U, const int* ldu,
                              float* const* S, int count) {
    return HWY_DYNAMIC_DISPATCH(SvdEconomy8BatchU)(m, A, lda, U, ldu, S, count);
}
#endif

}  // namespace nss
#endif
