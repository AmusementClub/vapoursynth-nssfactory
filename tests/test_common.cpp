// SPDX-License-Identifier: GPL-2.0-only
#include "nss/cpu_api.hpp"
#include "nss/cpu_common.hpp"
#include "hq_test_target.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <limits>
#include <vector>

static int fail(const char* msg) {
    std::fprintf(stderr, "%s\n", msg);
    return 1;
}

static int check_fused_admm_stationarity() {
    // Check the dual equation and the weighted least-squares stationarity
    // equation in FP64, independently of the production update expression.
    for (int m : {1, 7, 8, 9, 17, 48, 192, 243, 256}) {
        for (int n : {1, 4, 8, 17, 32}) {
            for (float penalty : {0.01f, 3.f, 9999.f}) {
                const int ldy = m + 3;
                constexpr float sentinel = -123.f;
                std::vector<float> y(ldy * n, sentinel), z(m * n), a(m * n + 2, sentinel),
                    x(m * n + 2, sentinel), w(m);
                for (int i = 0; i < m; ++i) w[i] = i % 3 == 0 ? 1.f : i % 3 == 1 ? 0.0625f : 1e-6f;
                for (int j = 0; j < n; ++j) {
                    for (int i = 0; i < m; ++i) {
                        const int p = j * m + i;
                        y[j * ldy + i] = float((p * 7) % 31 - 15) / 16.f;
                        z[p] = float((p * 3) % 17 - 8) / 16.f;
                        a[p + 1] = float(p % 7 - 3) / 32.f;
                        x[p + 1] = float((p * 5) % 19 - 9) / 16.f;
                    }
                }
                const auto old_a = a, old_x = x, old_y = y, old_z = z, old_w = w;
                const float next = std::min(10000.f, 1.7f * penalty);
                nss::admm_dual_add_weighted_x(a.data() + 1, x.data() + 1, y.data(), ldy, z.data(), w.data(),
                                             m, n, penalty, next);
                for (int j = 0; j < n; ++j) {
                    for (int i = 0; i < m; ++i) {
                        const int p = j * m + i;
                        const double dual_step = double(penalty) * (double(old_x[p + 1]) - z[p]);
                        const double expected_a = old_a[p + 1] + dual_step;
                        const double expected_x = (double(w[i]) * y[j * ldy + i] +
                                                   double(next) * 0.5 * z[p] - expected_a * 0.5) /
                                                  (double(w[i]) + double(next) * 0.5);
                        if (!std::isfinite(a[p + 1]) || !std::isfinite(x[p + 1]) ||
                            std::abs(a[p + 1] - expected_a) > 2e-6 * (1 + std::abs(expected_a)) ||
                            std::abs(x[p + 1] - expected_x) > 2e-6 * (1 + std::abs(expected_x))) {
                            return fail("fused ADMM violates dual or weighted-X stationarity");
                        }
                    }
                }
                if (a.front() != sentinel || a.back() != sentinel || x.front() != sentinel ||
                    x.back() != sentinel || y != old_y || z != old_z || w != old_w) {
                    return fail("fused ADMM modified padding or a read-only input");
                }
            }
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    if (!hq_test_target(argc, argv)) return 77;
    if (check_fused_admm_stationarity()) return 1;
    {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        const float inf = std::numeric_limits<float>::infinity();
        if (!nss::is_finite_bits(0.f) || nss::is_finite_bits(-inf) || nss::is_finite_bits(nan) ||
            nss::is_finite_bits(-nan) || nss::is_finite_bits(inf)) {
            return fail("is_finite_bits classification");
        }
    }

    float S[4] = {10.f, 4.f, 1.f, 0.5f};
    const int k = nss::sv_shrink(S, 4, 4.f, 1);
    if (k != 2) {
        return fail("sv_shrink kept count");
    }
    if (std::fabs(S[0] - 10.f) > 1e-6f) {
        return fail("sv_shrink must protect S[0] when start_k=1");
    }
    if (!(S[1] > 0.f && S[1] < 4.f)) {
        return fail("sv_shrink AC");
    }

    {
        // ClosedWNNM: σ̂ = (s + sqrt(s² − C))/2 while s² > C.
        float Cwn[4] = {5.f, 3.f, 1.2f, 0.4f};
        const float C = 2.f;
        const int kept = nss::sv_shrink(Cwn, 4, C, 0);
        if (kept != 2) {
            return fail("ClosedWNNM kept");
        }
        const float s0 = 5.f;
        const float expect0 = 0.5f * (s0 + std::sqrt(s0 * s0 - C));
        if (std::fabs(Cwn[0] - expect0) > 1e-5f) {
            return fail("ClosedWNNM formula");
        }
        if (std::fabs(Cwn[2] - 1.2f) > 1e-6f || std::fabs(Cwn[3] - 0.4f) > 1e-6f) {
            return fail("ClosedWNNM must stop and leave the tail");
        }
    }

    float g[8] = {1.f, 3.f, 5.f, 7.f, 2.f, 4.f, 6.f, 8.f};
    float mean[4];
    nss::group_center_sub(g, 4, 2, 4, mean);
    if (std::fabs(mean[0] - 1.5f) > 1e-6f || std::fabs(g[0] + 0.5f) > 1e-6f) {
        return fail("group_center_sub");
    }
    nss::group_center_add(g, 4, 2, 4, mean);
    if (std::fabs(g[0] - 1.f) > 1e-6f) {
        return fail("group_center_add");
    }

    float x[3] = {2.f, -0.5f, 0.25f};
    nss::soft_threshold(x, 3, 1.f);
    if (std::fabs(x[0] - 1.f) > 1e-6f || x[1] != 0.f || x[2] != 0.f) {
        return fail("soft_threshold");
    }

    float cur[2] = {0.f, 1.f};
    const float y[2] = {4.f, 5.f};
    nss::iter_regularize(cur, y, 2, 0.5f);
    if (std::fabs(cur[0] - 2.f) > 1e-6f || std::fabs(cur[1] - 3.f) > 1e-6f) {
        return fail("iter_regularize");
    }

    std::vector<float> plane(64, 0.f);
    for (int i = 0; i < 64; ++i) {
        plane[static_cast<std::size_t>(i)] = static_cast<float>(i);
    }
    const float* srcs[1] = {plane.data()};
    int strides[1] = {8};
    float col[64];
    nss::pack_patch_nch(col, 64, srcs, strides, 1, 0, 0, 8, 8, 8);
    if (std::fabs(col[0] - 0.f) > 1e-6f || std::fabs(col[9] - 9.f) > 1e-6f) {
        return fail("pack_patch_nch");
    }

    {
        std::vector<float> p0(16, 1.f);
        std::vector<float> p1(16, 2.f);
        std::vector<float> p2(16, 3.f);
        const float* ch[3] = {p0.data(), p1.data(), p2.data()};
        int st[3] = {4, 4, 4};
        float packed[48];
        nss::pack_patch_nch(packed, 48, ch, st, 3, 0, 0, 4, 4, 4);
        if (std::fabs(packed[0] - 1.f) > 1e-6f || std::fabs(packed[16] - 2.f) > 1e-6f ||
            std::fabs(packed[32] - 3.f) > 1e-6f) {
            return fail("pack_patch_nch nch=3");
        }
    }

    {
        float x[17];
        for (int i = 0; i < 17; ++i) {
            x[i] = (i % 2 == 0) ? 2.f : -0.5f;
        }
        nss::soft_threshold(x, 17, 1.f);
        if (std::fabs(x[0] - 1.f) > 1e-6f || x[1] != 0.f || std::fabs(x[16] - 1.f) > 1e-6f) {
            return fail("soft_threshold remainder");
        }
        float xv[4] = {2.f, -0.5f, 0.25f, -3.f};
        const float tv[4] = {1.f, 1.f, 1.f, 0.5f};
        nss::soft_threshold_var(xv, tv, 4);
        if (std::fabs(xv[0] - 1.f) > 1e-6f || xv[1] != 0.f || xv[2] != 0.f || std::fabs(xv[3] + 2.5f) > 1e-6f) {
            return fail("soft_threshold_var");
        }
        float bad_thresholds[4] = {std::numeric_limits<float>::quiet_NaN(),
                                   -std::numeric_limits<float>::quiet_NaN(),
                                   std::numeric_limits<float>::infinity(),
                                   -std::numeric_limits<float>::infinity()};
        float xt[4] = {2.f, -2.f, 3.f, -3.f};
        nss::soft_threshold_var(xt, bad_thresholds, 4);
        if (xt[0] != 2.f || xt[1] != -2.f || xt[2] != 3.f || xt[3] != -3.f) {
            return fail("soft_threshold_var non-finite threshold guard");
        }
        float xs = 2.f;
        nss::soft_threshold(&xs, 1, std::numeric_limits<float>::infinity());
        if (xs != 2.f) {
            return fail("soft_threshold non-finite threshold guard");
        }
        float a[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        const float b[9] = {2, 2, 2, 2, 2, 2, 2, 2, 2};
        nss::iter_regularize(a, b, 9, 0.5f);
        if (std::fabs(a[8] - 1.f) > 1e-6f) {
            return fail("iter_regularize remainder");
        }
    }

    {
        float Y[4] = {1.f, 1.f, 1.f, 1.f};
        float Z[4] = {0.f, 0.f, 0.f, 0.f};
        float A[4] = {0.f, 0.f, 0.f, 0.f};
        float w2[4] = {1.f, 1.f, 1.f, 1.f};
        float X[4] = {0.f, 0.f, 0.f, 0.f};
        nss::admm_weighted_x(X, Y, 4, Z, A, w2, 4, 1, 2.f);
        if (std::fabs(X[0] - 0.5f) > 1e-5f) {
            return fail("admm_weighted_x");
        }
        for (const float bad_rho : {std::numeric_limits<float>::quiet_NaN(),
                                    -std::numeric_limits<float>::quiet_NaN(),
                                    std::numeric_limits<float>::infinity(),
                                    -std::numeric_limits<float>::infinity()}) {
            std::fill(std::begin(X), std::end(X), 7.f);
            nss::admm_weighted_x(X, Y, 4, Z, A, w2, 4, 1, bad_rho);
            if (std::any_of(std::begin(X), std::end(X), [](float value) { return value != 7.f; })) {
                return fail("admm_weighted_x non-finite rho guard");
            }
        }
        float sigma[2] = {2.f, 4.f};
        float ww[8];
        const float smin = nss::channel_weight_diag(ww, 8, 2, sigma);
        if (std::fabs(smin - 2.f) > 1e-6f || std::fabs(ww[0] - 1.f) > 1e-6f || std::fabs(ww[4] - 0.25f) > 1e-6f) {
            return fail("channel_weight_diag");
        }

        const float bad_sigma[3] = {std::numeric_limits<float>::quiet_NaN(),
                                    -std::numeric_limits<float>::infinity(), 2.f};
        float bad_w[12]{};
        const float bad_min = nss::channel_weight_diag(bad_w, 12, 3, bad_sigma);
        if (std::fabs(bad_min - 2.f) > 1e-6f || !nss::is_finite_bits(bad_w[0]) || !nss::is_finite_bits(bad_w[4]) ||
            !nss::is_finite_bits(bad_w[8])) {
            return fail("channel_weight_diag non-finite sigma guard");
        }
    }

    {
        float a[7];
        float b[7];
        for (int i = 0; i < 7; ++i) {
            a[i] = static_cast<float>(i);
            b[i] = static_cast<float>(i + 1);
        }
        const float d = nss::dot_n(a, b, 7);
        if (std::fabs(d - 112.f) > 1e-3f) {
            return fail("dot_n remainder");
        }
    }

    {
        constexpr int w = 4;
        constexpr int h = 2;
        constexpr int sstride = 8;
        float src[16];
        float num[8];
        float den[8];
        float dst[8];
        for (int i = 0; i < 16; ++i) {
            src[i] = 99.f;
        }
        src[0] = 10.f;
        src[1] = 11.f;
        src[2] = 12.f;
        src[3] = 13.f;
        src[8] = 20.f;
        src[9] = 21.f;
        src[10] = 22.f;
        src[11] = 23.f;
        for (int i = 0; i < 8; ++i) {
            num[i] = 0.f;
            den[i] = 0.f;
            dst[i] = -1.f;
        }
        nss::aggregate_finish(dst, num, den, src, w, h, w, w, sstride);
        if (std::fabs(dst[0] - 10.f) > 1e-6f || std::fabs(dst[4] - 20.f) > 1e-6f ||
            std::fabs(dst[7] - 23.f) > 1e-6f) {
            return fail("aggregate_finish sstride fallback");
        }
    }

    std::printf("test_common ok\n");
    return 0;
}
