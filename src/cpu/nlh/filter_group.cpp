#include "nss/cpu_nlh.hpp"
#include "nss/params.hpp"
#include "cpu/hwy_config.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/nlh/filter_group.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

#include "cpu/nlh/haar_inl.hpp"

constexpr int kMaxM = 256;
constexpr int kMaxN = 16;
constexpr int kMaxQ = 8;

static int clamp_q(int q, int m) {
    if (q != 2 && q != 4 && q != 8) {
        q = kNlhDefaultQ;
    }
    while (q > m) {
        q /= 2;
    }
    return q;
}

// Largest Haar length in {2,4,8,16} that is <= n. 0 if n < 2 (skip transform).
static int haar_len_le(int n) {
    if (n >= 16) {
        return 16;
    }
    if (n >= 8) {
        return 8;
    }
    if (n >= 4) {
        return 4;
    }
    if (n >= 2) {
        return 2;
    }
    return 0;
}

// Paper (7): zero the last two rows except column 0. q=2 keeps only column 0.
static int struct_row0(int q) {
    return q > 2 ? q - 2 : 0;
}

static void haar2d(float* mat, int q, int n, bool inverse) {
    Haar2dFast(mat, q, n, inverse);
}

static void gather_rows_fix(float* dst, const float* group, int lda, const int* rows, int q, int n) {
    const hn::ScalableTag<float> d;
    const hn::Rebind<int32_t, hn::ScalableTag<float>> di;
    const int N = static_cast<int>(hn::Lanes(d));
    if (q == 4 && n == 16 && N == 16) {
        const auto idx = hn::Mul(hn::Iota(di, 0), hn::Set(di, lda));
        hn::Vec<hn::ScalableTag<float>> r0 = hn::GatherIndex(d, group + rows[0], idx);
        hn::Vec<hn::ScalableTag<float>> r1 = hn::GatherIndex(d, group + rows[1], idx);
        hn::Vec<hn::ScalableTag<float>> r2 = hn::GatherIndex(d, group + rows[2], idx);
        hn::Vec<hn::ScalableTag<float>> r3 = hn::GatherIndex(d, group + rows[3], idx);
        hn::StoreInterleaved4(r0, r1, r2, r3, d, dst);
        return;
    }
    std::memset(dst, 0, static_cast<std::size_t>(q * n) * sizeof(float));
    float tmp[16];
    const auto vlda = hn::Set(di, lda);
    for (int j = 0; j < q; ++j) {
        const float* base = group + rows[j];
        int c = 0;
        for (; c + N <= n; c += N) {
            const auto idx = hn::Mul(hn::Iota(di, static_cast<int32_t>(c)), vlda);
            hn::StoreU(hn::GatherIndex(d, base, idx), d, tmp + c);
        }
        for (; c < n; ++c) {
            tmp[c] = base[c * lda];
        }
        for (int cc = 0; cc < n; ++cc) {
            dst[j + cc * q] = tmp[cc];
        }
    }
}

static void bi_hard_4x16(float* y, float thr, int* kept) {
    Haar2d_4x16(y, false);
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const float dc = y[0];
    if (N == 16) {
        const auto vt = hn::Set(d, thr);
        const auto z = hn::Zero(d);
        for (int i = 0; i < 64; i += 16) {
            const auto v = hn::LoadU(d, y + i);
            hn::StoreU(hn::IfThenElse(hn::Ge(hn::Abs(v), vt), v, z), d, y + i);
        }
    } else {
        for (int i = 0; i < 64; ++i) {
            if (std::fabs(y[i]) < thr) {
                y[i] = 0.f;
            }
        }
    }
    y[0] = dc;
    for (int c = 1; c < 16; ++c) {
        y[c * 4 + 2] = 0.f;
        y[c * 4 + 3] = 0.f;
    }
    ++*kept;
    for (int i = 1; i < 64; ++i) {
        *kept += (y[i] != 0.f);
    }
    Haar2d_4x16(y, true);
}

static void coeff_hard(float* mat, int q, int n, float thr) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const auto vt = hn::Set(d, thr);
    const auto z = hn::Zero(d);
    const float dc = mat[0];
    for (int c = 0; c < n; ++c) {
        float* col = mat + c * q;
        int k = 0;
        for (; k + N <= q; k += N) {
            const auto v = hn::LoadU(d, col + k);
            hn::StoreU(hn::IfThenElse(hn::Ge(hn::Abs(v), vt), v, z), d, col + k);
        }
        for (; k < q; ++k) {
            if (std::fabs(col[k]) < thr) {
                col[k] = 0.f;
            }
        }
    }
    mat[0] = dc;
}

static void structural_hard(float* mat, int q, int n) {
    const int i0 = struct_row0(q);
    for (int c = 1; c < n; ++c) {
        for (int i = i0; i < q; ++i) {
            mat[i + c * q] = 0.f;
        }
    }
}

static void count_kept(const float* mat, int q, int n, int* kept) {
    ++*kept;
    for (int c = 0; c < n; ++c) {
        for (int i = 0; i < q; ++i) {
            if (i == 0 && c == 0) {
                continue;
            }
            if (mat[i + c * q] != 0.f) {
                ++*kept;
            }
        }
    }
}

static void wiener_ac(float* y, const float* r, int q, int n, float sig2, float* w2sum) {
    const int i0 = struct_row0(q);
    for (int c = 0; c < n; ++c) {
        for (int i = 0; i < q; ++i) {
            float w = 1.f;
            if (i == 0 && c == 0) {
                w = 1.f;
            } else if (c >= 1 && i >= i0) {
                w = 0.f;
            } else {
                const float rr = r[c * q + i];
                const float r2 = rr * rr;
                w = r2 / (r2 + sig2);
            }
            y[c * q + i] *= w;
            *w2sum += w * w;
        }
    }
}

static void bi_hard_2d(float* y, int q, int n, float thr, int* kept) {
    haar2d(y, q, n, false);
    coeff_hard(y, q, n, thr);
    structural_hard(y, q, n);
    count_kept(y, q, n, kept);
    haar2d(y, q, n, true);
}

void NlhFilterGroup(float* patches, int m, int n, int lda, int q, float sigma, bool wiener, const float* ref_patches,
                    float* weight_out, float* work, int work_floats) {
    if (!patches || m < 1 || n < 1 || lda < m) {
        if (weight_out) {
            *weight_out = 1.f;
        }
        return;
    }
    m = std::min(m, kMaxM);
    n = std::min(n, kMaxN);
    q = clamp_q(q, m);
    const int n_use = haar_len_le(n);
    if (q < 2 || n_use < 2) {
        if (weight_out) {
            *weight_out = 1.f;
        }
        return;
    }
    const int snap = lda * n;
    const int yoff = snap;
    const int roff = snap + q * n_use;
    const int idx_off = roff + q * n_use;
    const int idx_f = (m * q * static_cast<int>(sizeof(int)) + static_cast<int>(sizeof(float)) - 1) /
                      static_cast<int>(sizeof(float));
    const int need = idx_off + idx_f;
    float stack_y[kMaxQ * kMaxN];
    float stack_r[kMaxQ * kMaxN];
    std::vector<float> snap_store;
    std::vector<int> idx_store;
    float* src = nullptr;
    float* y = stack_y;
    float* rbuf = stack_r;
    int* idx = nullptr;
    if (work && work_floats >= need) {
        src = work;
        idx = reinterpret_cast<int*>(work + idx_off);
    } else {
        snap_store.resize(static_cast<std::size_t>(snap));
        src = snap_store.data();
        idx_store.resize(static_cast<std::size_t>(m * q));
        idx = idx_store.data();
    }
    for (int j = 0; j < n; ++j) {
        std::memcpy(src + j * lda, patches + j * lda, static_cast<std::size_t>(m) * sizeof(float));
    }

    const float* match_src = (wiener && ref_patches) ? ref_patches : src;
    pixel_match(match_src, m, n, lda, q, idx);

    // Orthonormal Haar: coefficient noise std is σ, not σ².
    const float thr = kBmHardLambda * std::fabs(sigma);
    const float sig2 = sigma * sigma;
    int kept = 0;
    float w2sum = 0.f;

    const bool fast416 = (q == 4 && n_use == 16 && !(wiener && ref_patches));
    for (int row = 0; row < m; ++row) {
        const int* rows = idx + row * q;
        gather_rows_fix(y, src, lda, rows, q, n_use);
        if (wiener && ref_patches) {
            gather_rows_fix(rbuf, ref_patches, lda, rows, q, n_use);
            haar2d(y, q, n_use, false);
            haar2d(rbuf, q, n_use, false);
            wiener_ac(y, rbuf, q, n_use, sig2, &w2sum);
            haar2d(y, q, n_use, true);
        } else if (fast416) {
            bi_hard_4x16(y, thr, &kept);
        } else {
            bi_hard_2d(y, q, n_use, thr, &kept);
        }
        for (int c = 0; c < n_use; ++c) {
            patches[row + c * lda] = y[c * q];
        }
    }

    if (weight_out) {
        if (wiener) {
            *weight_out = 1.f / std::max(w2sum, 1e-12f);
        } else {
            *weight_out = 1.f / static_cast<float>(std::max(kept, 1));
        }
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(NlhFilterGroup);

void nlh_filter_group(float* patches, int m, int n, int lda, int q, float sigma, bool wiener,
                      const float* ref_patches, float* weight_out, float* work, int work_floats) {
    HWY_DYNAMIC_DISPATCH(NlhFilterGroup)(patches, m, n, lda, q, sigma, wiener, ref_patches, weight_out, work,
                                         work_floats);
}

}  // namespace nss
#endif
