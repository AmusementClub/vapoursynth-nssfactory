#include "nss/cpu_api.hpp"
#include "nss/cpu_common.hpp"
#include "nss/cpu_mcwnnm.hpp"
#include "nss/params.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

static int fail(const char* msg) {
    std::fprintf(stderr, "%s\n", msg);
    return 1;
}

int main() {
    constexpr int nch = 3;
    constexpr int block = 4;
    constexpr int n = 4;
    constexpr int m = nch * block * block;
    constexpr int lda = m;
    std::vector<float> Y(static_cast<std::size_t>(lda * n));
    double mean_in = 0.0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            const float v = 0.5f + 0.01f * static_cast<float>((i + j) % 3);
            Y[static_cast<std::size_t>(i + j * lda)] = v;
            mean_in += static_cast<double>(v);
        }
    }
    mean_in /= static_cast<double>(m * n);

    const float sigma[3] = {nss::kMcwnnmDefaultSigma / 255.f, nss::kMcwnnmDefaultSigma / 255.f,
                            (nss::kMcwnnmDefaultSigma * 1.5f) / 255.f};
    const int work_n = nss::mcwnnm_filter_work_floats(m, n);
    std::vector<float> work(static_cast<std::size_t>(work_n));
    float aw = 0.f;
    if (nss::mcwnnm_filter_group(Y.data(), m, n, lda, nch, sigma, nss::kMcwnnmDefaultAdmmIter, nss::kMcwnnmDefaultRho,
                                 nss::kMcwnnmDefaultMu, 0, 1, &aw, work.data(), work_n) != 0) {
        return fail("mcwnnm_filter_group failed");
    }

    double mean_out = 0.0;
    int nonfinite = 0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            const float v = Y[static_cast<std::size_t>(i + j * lda)];
            if (!std::isfinite(v)) {
                ++nonfinite;
            }
            mean_out += static_cast<double>(v);
        }
    }
    mean_out /= static_cast<double>(m * n);
    std::printf("mcwnnm finite=%d mean_in=%.6g mean_out=%.6g aw=%.6g\n", nonfinite == 0, mean_in, mean_out,
                static_cast<double>(aw));
    if (nonfinite != 0) {
        return fail("ADMM produced non-finite values");
    }
    if (std::fabs(mean_out - mean_in) > 0.05) {
        return fail("DC not preserved");
    }
    if (!(aw > 0.f) || !std::isfinite(aw)) {
        return fail("adaptive weight");
    }

    {
        constexpr int lda_pad = m + 16;
        std::vector<float> Yp(static_cast<std::size_t>(lda_pad * n), 0.f);
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                Yp[static_cast<std::size_t>(i + j * lda_pad)] = 0.2f + 0.01f * static_cast<float>(i);
            }
        }
        if (nss::mcwnnm_filter_group(Yp.data(), m, n, lda_pad, nch, sigma, 4, 1.f, 1.f, 1, 0, nullptr, work.data(),
                                     work_n) != 0) {
            return fail("mcwnnm padded lda failed");
        }
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                if (!std::isfinite(Yp[static_cast<std::size_t>(i + j * lda_pad)])) {
                    return fail("padded lda non-finite");
                }
            }
        }
    }

    {
        std::vector<float> Y0(static_cast<std::size_t>(lda * n), 0.31f);
        const std::vector<float> orig = Y0;
        const float zsig[3] = {0.f, 0.f, 0.f};
        float aw0 = 0.f;
        if (nss::mcwnnm_filter_group(Y0.data(), m, n, lda, nch, zsig, 2, 1.f, 1.f, 0, 1, &aw0, work.data(), work_n) !=
            0) {
            return fail("mcwnnm sigma=0 failed");
        }
        if (std::fabs(aw0 - 1.f) > 1e-6f) {
            return fail("sigma=0 adaptive weight should be 1");
        }
        for (int i = 0; i < lda * n; ++i) {
            if (std::fabs(Y0[static_cast<std::size_t>(i)] - orig[static_cast<std::size_t>(i)]) > 1e-6f) {
                return fail("sigma=0 should be identity");
            }
        }
    }

    {
        // Invalid channel noise is handled by the shared finite guard rather
        // than being allowed to contaminate the Gram/QR path. The valid
        // channel still supplies the scale and the operation remains finite.
        std::vector<float> Ybad(static_cast<std::size_t>(lda * n), 0.21f);
        const float bad_sigma[3] = {std::numeric_limits<float>::quiet_NaN(),
                                    std::numeric_limits<float>::infinity(), sigma[2]};
        float aw_bad = 0.f;
        if (nss::mcwnnm_filter_group(Ybad.data(), m, n, lda, nch, bad_sigma, 2, 1.f, 1.f, 0, 1, &aw_bad,
                                     work.data(), work_n) != 0) {
            return fail("mcwnnm non-finite sigma fallback failed");
        }
        for (float value : Ybad) {
            if (!nss::is_finite_bits(value)) {
                return fail("mcwnnm non-finite sigma produced non-finite output");
            }
        }
    }

    {
        constexpr int gm = 192;
        constexpr int gn = 8;
        std::vector<float> gram_group(static_cast<std::size_t>(gm * gn), 0.f);
        for (int col = 0; col < gn; ++col) {
            for (int row = 0; row < gm; ++row) {
                gram_group[static_cast<std::size_t>(row + col * gm)] =
                    0.1f + 0.001f * static_cast<float>((row + 3 * col) % 17);
            }
        }
        const int gram_work_n = nss::mcwnnm_filter_work_floats(gm, gn);
        std::vector<float> gram_work(static_cast<std::size_t>(gram_work_n));
        const float gram_sigma[3] = {sigma[0], std::numeric_limits<float>::infinity(), sigma[2]};
        if (nss::mcwnnm_filter_group(gram_group.data(), gm, gn, gm, nch, gram_sigma, 2, 1.f, 1.f, 0, 0, nullptr,
                                     gram_work.data(), gram_work_n) != 0) {
            return fail("mcwnnm Gram path failed with non-finite sigma");
        }
        for (float value : gram_group) {
            if (!nss::is_finite_bits(value)) {
                return fail("mcwnnm Gram path produced non-finite output");
            }
        }
    }

    std::vector<float> plane(static_cast<std::size_t>(block * block), 0.4f);
    const float* srcs[3] = {plane.data(), plane.data(), plane.data()};
    int strides[3] = {block, block, block};
    std::vector<float> col(static_cast<std::size_t>(m));
    nss::pack_patch_nch(col.data(), m, srcs, strides, nch, 0, 0, block, block, block);
    if (std::fabs(col[0] - 0.4f) > 1e-6f || std::fabs(col[block * block] - 0.4f) > 1e-6f) {
        return fail("pack_patch_nch");
    }

    {
        std::vector<float> lo(static_cast<std::size_t>(lda * n));
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                lo[static_cast<std::size_t>(i + j * lda)] = 0.5f + 0.02f * static_cast<float>((i + 3 * j) % 5);
            }
        }
        std::vector<float> hi = lo;
        const float slo[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
        const float shi[3] = {20.f / 255.f, 20.f / 255.f, 20.f / 255.f};
        if (nss::mcwnnm_filter_group(lo.data(), m, n, lda, nch, slo, 4, 1.f, 1.f, 0, 0, nullptr, work.data(),
                                     work_n) != 0 ||
            nss::mcwnnm_filter_group(hi.data(), m, n, lda, nch, shi, 4, 1.f, 1.f, 0, 0, nullptr, work.data(),
                                     work_n) != 0) {
            return fail("mcwnnm sigma sweep");
        }
        double e_lo = 0.0;
        double e_hi = 0.0;
        for (int i = 0; i < lda * n; ++i) {
            e_lo += static_cast<double>(lo[static_cast<std::size_t>(i)]) * lo[static_cast<std::size_t>(i)];
            e_hi += static_cast<double>(hi[static_cast<std::size_t>(i)]) * hi[static_cast<std::size_t>(i)];
        }
        std::printf("mcwnnm energy lo=%.6g hi=%.6g\n", e_lo, e_hi);
        if (!(e_hi <= e_lo + 1e-6)) {
            return fail("larger sigma should not increase energy");
        }
    }

    {
        std::vector<float> Ym(static_cast<std::size_t>(lda * n));
        double mean_in = 0.0;
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                const float v = 0.4f + 0.02f * static_cast<float>((i + j) % 4);
                Ym[static_cast<std::size_t>(i + j * lda)] = v;
                mean_in += static_cast<double>(v);
            }
        }
        mean_in /= static_cast<double>(m * n);
        float aw = 0.f;
        if (nss::mcwnnm_filter_group(Ym.data(), m, n, lda, nch, sigma, 4, nss::kMcwnnmDefaultRho,
                                     nss::kMcwnnmDefaultMu, 1, 0, &aw, work.data(), work_n) != 0) {
            return fail("mcwnnm residual=1 failed");
        }
        double mean_out = 0.0;
        for (int i = 0; i < lda * n; ++i) {
            mean_out += static_cast<double>(Ym[static_cast<std::size_t>(i)]);
        }
        mean_out /= static_cast<double>(m * n);
        std::printf("mcwnnm residual=1 mean_in=%.6g mean_out=%.6g aw=%.6g\n", mean_in, mean_out,
                    static_cast<double>(aw));
        if (std::fabs(mean_out - mean_in) > 0.05) {
            return fail("residual=1 must restore row means");
        }
        if (std::fabs(aw - 1.f) > 1e-6f) {
            return fail("adaptive=0 weight must be 1");
        }
    }

    {
        std::vector<float> Yb(static_cast<std::size_t>(lda * n), 0.2f);
        if (nss::mcwnnm_filter_group(Yb.data(), 300, n, 300, nch, sigma, 2, 1.f, 1.001f, 1, 0, nullptr, work.data(),
                                     work_n) != -1) {
            return fail("mcwnnm m>kSvdMaxM must fail before centering");
        }
        if (nss::mcwnnm_filter_group(Yb.data(), m, n, lda, nch, sigma, 2, 1.f, 1.001f, 1, 0, nullptr, work.data(),
                                     4) != -1) {
            return fail("mcwnnm short work must fail");
        }
    }

    for (int gm : {12, 48, 108, 147, 192, 243}) {
        constexpr int gn = 8;
        const int glda = (gm + 15) & ~15;
        const int gwork_n = nss::mcwnnm_filter_work_floats(gm, gn);
        std::vector<float> input(static_cast<std::size_t>(glda * gn), 0.f);
        for (int col = 0; col < gn; ++col) {
            for (int row = 0; row < gm; ++row) {
                input[static_cast<std::size_t>(row + col * glda)] =
                    0.35f + 0.07f * std::sin(static_cast<float>((row + 1) * (col + 2)) * 0.031f) +
                    0.01f * static_cast<float>(col);
            }
        }
        std::vector<float> first = input;
        std::vector<float> repeat = input;
        std::vector<float> first_work(static_cast<std::size_t>(gwork_n));
        std::vector<float> repeat_work(static_cast<std::size_t>(gwork_n));
        const float gram_sigma[3] = {0.25f / 255.f, 3.f / 255.f, 12.f / 255.f};
        if (nss::mcwnnm_filter_group(first.data(), gm, gn, glda, 3, gram_sigma, 10,
                                     nss::kMcwnnmDefaultRho, nss::kMcwnnmDefaultMu, 1, 0, nullptr,
                                     first_work.data(), gwork_n) != 0 ||
            nss::mcwnnm_filter_group(repeat.data(), gm, gn, glda, 3, gram_sigma, 10,
                                     nss::kMcwnnmDefaultRho, nss::kMcwnnmDefaultMu, 1, 0, nullptr,
                                     repeat_work.data(), gwork_n) != 0) {
            return fail("mcwnnm n=8 Gram path failed");
        }
        for (int col = 0; col < gn; ++col) {
            for (int row = 0; row < gm; ++row) {
                if (!std::isfinite(first[static_cast<std::size_t>(row + col * glda)])) {
                    return fail("mcwnnm n=8 Gram path produced non-finite output");
                }
            }
        }
        if (std::memcmp(first.data(), repeat.data(), first.size() * sizeof(float)) != 0) {
            return fail("mcwnnm n=8 Gram path is not deterministic");
        }
    }

    std::printf("test_mcwnnm ok\n");
    return 0;
}
