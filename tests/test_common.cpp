#include "nss/cpu_api.hpp"
#include "nss/cpu_common.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

static int fail(const char* msg) {
    std::fprintf(stderr, "%s\n", msg);
    return 1;
}

int main() {
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
        float sigma[2] = {2.f, 4.f};
        float ww[8];
        const float smin = nss::channel_weight_diag(ww, 8, 2, sigma);
        if (std::fabs(smin - 2.f) > 1e-6f || std::fabs(ww[0] - 1.f) > 1e-6f || std::fabs(ww[4] - 0.25f) > 1e-6f) {
            return fail("channel_weight_diag");
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
