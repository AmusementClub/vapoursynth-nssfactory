#include "cpu/wnnm/jacobi8.hpp"
#include "cpu/hwy_config.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/wnnm/svd_batch8.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

namespace {

constexpr int kBatchLanes = 16;
constexpr int kN = 8;

inline std::size_t TallIndex(int row, int col, int lane) {
    return (static_cast<std::size_t>(col) * kSvdBatch8MaxM + row) * kBatchLanes + lane;
}

inline std::size_t SmallIndex(int row, int col, int lane) {
    return (static_cast<std::size_t>(col) * kN + row) * kBatchLanes + lane;
}

template <bool kNeedVt, class D>
void SvdChunk(D d, int m, const float* const* A, const int* lda, float* const* U, const int* ldu,
              float* const* S, float* const* Vt, const int* ldvt, int count) {
    using V = hn::Vec<D>;
    using M = hn::Mask<D>;

    alignas(64) float tall[kSvdBatch8MaxM * kN * kBatchLanes];
    alignas(64) float reflectors[kSvdBatch8MaxM * kN * kBatchLanes];
    alignas(64) float rsoa[kN * kN * kBatchLanes];
    alignas(64) float vsoa[kN * kN * kBatchLanes];
    alignas(64) float ssoa[kN * kBatchLanes];
    alignas(64) float beta[kN * kBatchLanes];
    std::memset(tall, 0, sizeof(tall));
    std::memset(reflectors, 0, sizeof(reflectors));
    std::memset(rsoa, 0, sizeof(rsoa));
    std::memset(ssoa, 0, sizeof(ssoa));
    std::memset(beta, 0, sizeof(beta));

    for (int lane = 0; lane < count; ++lane) {
        for (int col = 0; col < kN; ++col) {
            for (int row = 0; row < m; ++row) {
                tall[TallIndex(row, col, lane)] = A[lane][row + col * lda[lane]];
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

        for (int col = k; col < kN; ++col) {
            V dot = zero;
            for (int row = k; row < m; ++row) {
                dot = hn::MulAdd(hn::Load(d, reflectors + TallIndex(row, k, 0)),
                                 hn::Load(d, tall + TallIndex(row, col, 0)), dot);
            }
            const V factor = hn::Mul(b, dot);
            for (int row = k; row < m; ++row) {
                const V value = hn::Load(d, tall + TallIndex(row, col, 0));
                const V updated = hn::NegMulAdd(factor, hn::Load(d, reflectors + TallIndex(row, k, 0)), value);
                hn::Store(updated, d, tall + TallIndex(row, col, 0));
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

    static constexpr int pairs[7][4][2] = {
        {{0, 1}, {2, 3}, {4, 5}, {6, 7}}, {{0, 2}, {1, 3}, {4, 6}, {5, 7}},
        {{0, 3}, {1, 2}, {4, 7}, {5, 6}}, {{0, 4}, {1, 5}, {2, 6}, {3, 7}},
        {{0, 5}, {1, 4}, {2, 7}, {3, 6}}, {{0, 6}, {1, 7}, {2, 4}, {3, 5}},
        {{0, 7}, {1, 6}, {2, 5}, {3, 4}},
    };
    M active = valid;
    for (int sweep = 0; sweep < 8; ++sweep) {
        V max_off = zero;
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
                max_off = hn::Max(max_off, hn::Abs(apq));
                const V threshold = hn::Mul(hn::Set(d, 1e-14f),
                                            hn::Add(hn::Sqrt(hn::Max(hn::Mul(app, aqq), zero)),
                                                    hn::Set(d, 1e-20f)));
                const M rotate = hn::And(active, hn::Gt(hn::Abs(apq), threshold));
                const V safe_apq = hn::IfThenElse(rotate, apq, one);
                const V zeta = hn::Div(hn::Sub(aqq, app), hn::Mul(hn::Set(d, 2.0f), safe_apq));
                const V t = hn::Div(hn::CopySign(one, zeta),
                                    hn::Add(hn::Abs(zeta), hn::Sqrt(hn::MulAdd(zeta, zeta, one))));
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
            }
            for (int col = 0; col < kN; ++col) {
                norms[col] = zero;
                for (int row = 0; row < kN; ++row) {
                    norms[col] = hn::MulAdd(ju[col * kN + row], ju[col * kN + row], norms[col]);
                }
            }
        }
        active = hn::And(active, hn::Ge(max_off, hn::Set(d, 1e-8f)));
    }

    for (int col = 0; col < kN; ++col) {
        const V singular = hn::Sqrt(hn::Max(norms[col], zero));
        const M nonzero = hn::And(valid, hn::Gt(singular, hn::Set(d, 1e-20f)));
        const V inv = hn::IfThenElse(nonzero, hn::Div(one, hn::IfThenElse(nonzero, singular, one)), one);
        hn::Store(singular, d, ssoa + static_cast<std::size_t>(col) * kBatchLanes);
        for (int row = 0; row < kN; ++row) {
            hn::Store(hn::IfThenElse(nonzero, hn::Mul(ju[col * kN + row], inv), ju[col * kN + row]), d,
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

    std::memset(tall, 0, sizeof(tall));
    for (int col = 0; col < kN; ++col) {
        for (int row = 0; row < kN; ++row) {
            hn::Store(hn::Load(d, rsoa + SmallIndex(row, col, 0)), d, tall + TallIndex(row, col, 0));
        }
    }
    for (int k = kN - 1; k >= 0; --k) {
        const V b = hn::Load(d, beta + static_cast<std::size_t>(k) * kBatchLanes);
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
    }

    for (int lane = 0; lane < count; ++lane) {
        for (int col = 0; col < kN; ++col) {
            S[lane][col] = ssoa[static_cast<std::size_t>(col) * kBatchLanes + lane];
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
        SvdChunk<true>(d, m, A + begin, lda + begin, U + begin, ldu + begin, S + begin, Vt + begin, ldvt + begin,
                       chunk);
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
        SvdChunk<false>(d, m, A + begin, lda + begin, U + begin, ldu + begin, S + begin, nullptr, nullptr, chunk);
    }
    return 0;
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(SvdEconomy8Batch);
HWY_EXPORT(SvdEconomy8BatchU);

int svd_economy_8_batch_hwy(int m, const float* const* A, const int* lda, float* const* U, const int* ldu,
                            float* const* S, float* const* Vt, const int* ldvt, int count) {
    return HWY_DYNAMIC_DISPATCH(SvdEconomy8Batch)(m, A, lda, U, ldu, S, Vt, ldvt, count);
}

int svd_economy_8_batch_u_hwy(int m, const float* const* A, const int* lda, float* const* U, const int* ldu,
                              float* const* S, int count) {
    return HWY_DYNAMIC_DISPATCH(SvdEconomy8BatchU)(m, A, lda, U, ldu, S, count);
}

}  // namespace nss
#endif
