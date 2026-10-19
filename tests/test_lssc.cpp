// SPDX-License-Identifier: GPL-2.0-only
#include "nss/cpu_api.hpp"
#include "nss/cpu_common.hpp"
#include "nss/cpu_lssc.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

static int fail(const char* msg) {
    std::fprintf(stderr, "%s\n", msg);
    return 1;
}

static int test_cluster_assigns_every_patch() {
    constexpr int m = 16;
    constexpr int n = 20;
    constexpr int lda = 16;
    constexpr int k = nss::kLsscDefaultClusters;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> patches(static_cast<std::size_t>(lda * n), 0.f);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            patches[static_cast<std::size_t>(i + j * lda)] = dist(rng) + 0.25f * static_cast<float>(j % 4);
        }
    }
    std::vector<int> assign(static_cast<std::size_t>(n), -1);
    std::vector<int> counts(static_cast<std::size_t>(k), 0);
    nss::lssc_cluster(patches.data(), m, n, lda, k, assign.data(), counts.data());
    int sum = 0;
    for (int c = 0; c < k; ++c) {
        if (counts[static_cast<std::size_t>(c)] < 0) {
            return fail("cluster negative count");
        }
        sum += counts[static_cast<std::size_t>(c)];
    }
    if (sum != n) {
        return fail("cluster counts do not cover every patch");
    }
    for (int i = 0; i < n; ++i) {
        const int a = assign[static_cast<std::size_t>(i)];
        if (a < 0 || a >= k) {
            return fail("cluster left a patch unassigned");
        }
    }

    constexpr int k3 = 3;
    std::fill(assign.begin(), assign.end(), -1);
    std::vector<int> counts3(k3, 0);
    nss::lssc_cluster(patches.data(), m, n, lda, k3, assign.data(), counts3.data());
    int sum3 = 0;
    for (int c = 0; c < k3; ++c) {
        sum3 += counts3[static_cast<std::size_t>(c)];
    }
    if (sum3 != n) {
        return fail("k=3 cluster missed patches");
    }
    for (int i = 0; i < n; ++i) {
        if (assign[static_cast<std::size_t>(i)] < 0 || assign[static_cast<std::size_t>(i)] >= k3) {
            return fail("k=3 unassigned patch");
        }
    }
    std::printf("cluster assigned n=%d k=%d and k=%d\n", n, k, k3);
    return 0;
}

static int test_reconstruct_finite() {
    constexpr int block = 8;
    constexpr int m = block * block;
    constexpr int n = 12;
    constexpr int atoms = 16;
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(0.1f, 0.9f);
    std::vector<float> patches(static_cast<std::size_t>(m * n));
    for (float& v : patches) {
        v = dist(rng);
    }
    std::vector<float> D(static_cast<std::size_t>(m * atoms), 0.f);
    nss::lssc_dict_init(D.data(), m, atoms, m, patches.data(), n, m, block, 1, 123u);
    for (int a = 0; a < atoms; ++a) {
        for (int i = 0; i < m; ++i) {
            if (!std::isfinite(D[static_cast<std::size_t>(i + a * m)])) {
                return fail("dict produced non-finite atom");
            }
        }
    }

    // The explicit dictionary and OMP contracts must be usable at their
    // advertised exact sizes. This also exercises the K-SVD OMP scratch that
    // is carved out of the caller-owned dictionary workspace.
    std::vector<float> D_ws(static_cast<std::size_t>(m * atoms), 0.f);
    const int dict_need = nss::lssc_dict_work_floats(m, atoms, n, 1);
    std::vector<float> dict_work(static_cast<std::size_t>(dict_need), 0.f);
    nss::lssc_dict_init_workspace(D_ws.data(), m, atoms, m, patches.data(), n, m, block, 1, 123u, dict_work.data(),
                                  dict_need);
    for (std::size_t i = 0; i < D.size(); ++i) {
        if (!std::isfinite(D_ws[i]) || std::fabs(D_ws[i] - D[i]) > 4e-4f) {
            return fail("workspace dictionary differs from wrapper");
        }
    }

    std::vector<float> coeff_scalar(static_cast<std::size_t>(atoms), 0.f);
    std::vector<float> coeff_workspace(static_cast<std::size_t>(atoms), 0.f);
    std::vector<float> omp_work(static_cast<std::size_t>(nss::lssc_omp_work_floats(m, atoms, 8)), 0.f);
    const int omp_scalar = nss::lssc_omp(patches.data(), m, D.data(), atoms, m, 8, coeff_scalar.data());
    const int omp_workspace = nss::lssc_omp_workspace(patches.data(), m, D.data(), atoms, m, 8,
                                                       coeff_workspace.data(), omp_work.data(),
                                                       static_cast<int>(omp_work.size()));
    if (omp_scalar != omp_workspace || !std::equal(coeff_scalar.begin(), coeff_scalar.end(), coeff_workspace.begin(),
                                                    [](float a, float b) { return std::fabs(a - b) <= 2e-5f; })) {
        return fail("workspace OMP differs from wrapper");
    }

    std::fill(coeff_workspace.begin(), coeff_workspace.end(), 1.f);
    // A non-positive sparsity is a no-op in the legacy API. The workspace
    // overload must keep that behavior and must not require a scratch buffer.
    if (nss::lssc_omp_workspace(patches.data(), m, D.data(), atoms, m, 0, coeff_workspace.data(), nullptr, 0) != 0 ||
        std::any_of(coeff_workspace.begin(), coeff_workspace.end(), [](float value) { return value != 0.f; })) {
        return fail("workspace OMP changed non-positive sparsity behavior");
    }

    // The prepared frame path must be numerically equivalent to the legacy
    // one-shot API while reusing transpose/Lipschitz state.
    std::vector<float> legacy_input = patches;
    std::vector<float> prepared_input = patches;
    std::vector<float> legacy_work(static_cast<std::size_t>(nss::lssc_reconstruct_work_floats(m, n, atoms)));
    std::vector<float> prep_work(static_cast<std::size_t>(nss::lssc_prepare_work_floats(m, atoms)));
    std::vector<float> group_work(static_cast<std::size_t>(nss::lssc_reconstruct_prepared_work_floats(m, n, atoms)));
    nss::LsscPreparedContext context;
    if (nss::lssc_prepare_context(D.data(), m, atoms, m, prep_work.data(), static_cast<int>(prep_work.size()),
                                  &context) != 0) {
        return fail("lssc prepare context failed");
    }
    nss::lssc_reconstruct(legacy_input.data(), m, n, m, D.data(), atoms, m, 3.f / 255.f, legacy_work.data(),
                          static_cast<int>(legacy_work.size()));
    nss::lssc_reconstruct_prepared(prepared_input.data(), m, n, m, &context, 3.f / 255.f, group_work.data(),
                                   static_cast<int>(group_work.size()));
    for (std::size_t i = 0; i < legacy_input.size(); ++i) {
        if (std::fabs(legacy_input[i] - prepared_input[i]) > 3e-4f) {
            return fail("prepared reconstruction differs from legacy path");
        }
    }
    nss::lssc_reconstruct(patches.data(), m, n, m, D.data(), atoms, m, 3.f / 255.f);
    for (int i = 0; i < m * n; ++i) {
        if (!std::isfinite(patches[static_cast<std::size_t>(i)])) {
            return fail("reconstruct produced non-finite values");
        }
    }

    for (const float bad_sigma : {std::numeric_limits<float>::quiet_NaN(),
                                  -std::numeric_limits<float>::quiet_NaN(),
                                  std::numeric_limits<float>::infinity(),
                                  -std::numeric_limits<float>::infinity()}) {
        std::vector<float> bad_input = legacy_input;
        nss::lssc_reconstruct(bad_input.data(), m, n, m, D.data(), atoms, m, bad_sigma, legacy_work.data(),
                              static_cast<int>(legacy_work.size()));
        for (float value : bad_input) {
            if (!nss::is_finite_bits(value)) {
                return fail("lssc non-finite sigma produced non-finite output");
            }
        }
    }

    std::vector<float> A(static_cast<std::size_t>(atoms * n), 0.f);
    for (int i = 0; i < atoms * n; ++i) {
        A[static_cast<std::size_t>(i)] = dist(rng) - 0.5f;
    }
    nss::lssc_group_soft(A.data(), atoms, n, atoms, 0.25f);
    for (int i = 0; i < atoms * n; ++i) {
        if (!std::isfinite(A[static_cast<std::size_t>(i)])) {
            return fail("group_soft non-finite");
        }
    }

    std::vector<float> coef(static_cast<std::size_t>(atoms), 0.f);
    const int ksel = nss::lssc_omp(patches.data(), m, D.data(), atoms, m, 8, coef.data());
    if (ksel < 0) {
        return fail("omp failed");
    }
    for (int i = 0; i < atoms; ++i) {
        if (!std::isfinite(coef[static_cast<std::size_t>(i)])) {
            return fail("omp non-finite");
        }
    }
    std::printf("reconstruct finite atoms=%d omp_sel=%d\n", atoms, ksel);
    return 0;
}

static double mse(const float* a, const float* b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        const double d = static_cast<double>(a[i] - b[i]);
        s += d * d;
    }
    return s / static_cast<double>(n);
}

// Independent FP64 model of the cluster coding (include/nss/params/lssc.hpp):
// the preconditioner by a dense inverse of D D^T + delta I, the momentum from
// the FISTA t-sequence, the group soft threshold at lambda ||M d_a||. It takes
// only the power estimate L from lssc_prepare_context.
static void coding_reference(const std::vector<float>& input, int m, int n, const std::vector<float>& D, int atoms,
                             double lipschitz, double sigma, std::vector<double>& out) {
    std::vector<double> Y(static_cast<std::size_t>(m) * n), mean(n);
    for (int j = 0; j < n; ++j) {
        double s = 0.;
        for (int i = 0; i < m; ++i) s += input[static_cast<std::size_t>(j) * m + i];
        mean[j] = s / m;
        for (int i = 0; i < m; ++i) Y[static_cast<std::size_t>(j) * m + i] = input[static_cast<std::size_t>(j) * m + i] - mean[j];
    }
    const auto d = [&](int a, int i) { return static_cast<double>(D[static_cast<std::size_t>(a) * m + i]); };
    std::vector<double> S(static_cast<std::size_t>(m) * m, 0.);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
            for (int a = 0; a < atoms; ++a) S[static_cast<std::size_t>(i) * m + j] += d(a, i) * d(a, j);
    double trace = 0.;
    for (int i = 0; i < m; ++i) trace += S[static_cast<std::size_t>(i) * m + i];
    const double smean = trace / m, delta = nss::kLsscPrecondAlpha * smean, c = smean + delta;
    // M = c (S + delta I)^-1 by Gauss-Jordan elimination with partial pivoting.
    std::vector<double> A(S.begin(), S.end()), M(static_cast<std::size_t>(m) * m, 0.);
    for (int i = 0; i < m; ++i) {
        A[static_cast<std::size_t>(i) * m + i] += delta;
        M[static_cast<std::size_t>(i) * m + i] = c;
    }
    for (int k = 0; k < m; ++k) {
        int piv = k;
        for (int i = k + 1; i < m; ++i)
            if (std::fabs(A[static_cast<std::size_t>(i) * m + k]) > std::fabs(A[static_cast<std::size_t>(piv) * m + k])) piv = i;
        for (int j = 0; j < m; ++j) {
            std::swap(A[static_cast<std::size_t>(k) * m + j], A[static_cast<std::size_t>(piv) * m + j]);
            std::swap(M[static_cast<std::size_t>(k) * m + j], M[static_cast<std::size_t>(piv) * m + j]);
        }
        const double inv = 1. / A[static_cast<std::size_t>(k) * m + k];
        for (int j = 0; j < m; ++j) {
            A[static_cast<std::size_t>(k) * m + j] *= inv;
            M[static_cast<std::size_t>(k) * m + j] *= inv;
        }
        for (int i = 0; i < m; ++i) {
            if (i == k) continue;
            const double f = A[static_cast<std::size_t>(i) * m + k];
            for (int j = 0; j < m; ++j) {
                A[static_cast<std::size_t>(i) * m + j] -= f * A[static_cast<std::size_t>(k) * m + j];
                M[static_cast<std::size_t>(i) * m + j] -= f * M[static_cast<std::size_t>(k) * m + j];
            }
        }
    }
    std::vector<double> P(static_cast<std::size_t>(atoms) * m, 0.), weight(atoms, 0.);  // P[a][i] = (M d_a)_i
    for (int a = 0; a < atoms; ++a) {
        double norm2 = 0.;
        for (int i = 0; i < m; ++i) {
            double v = 0.;
            for (int j = 0; j < m; ++j) v += M[static_cast<std::size_t>(i) * m + j] * d(a, j);
            P[static_cast<std::size_t>(a) * m + i] = v;
            norm2 += v * v;
        }
        weight[a] = std::sqrt(norm2);
    }
    const double mu = 0.9 / (c * lipschitz / (lipschitz + delta));
    const double lam = mu * std::fabs(sigma) * std::sqrt(static_cast<double>(n));
    const std::size_t codes = static_cast<std::size_t>(atoms) * n;
    std::vector<double> X(codes, 0.), Xp(codes, 0.), Z(codes, 0.), G(codes, 0.), R(static_cast<std::size_t>(m) * n);
    double t = 1.;
    for (int it = 0; it < nss::kLsscCodingRounds; ++it) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                double v = Y[static_cast<std::size_t>(j) * m + i];
                for (int a = 0; a < atoms; ++a) v -= d(a, i) * Z[static_cast<std::size_t>(j) * atoms + a];
                R[static_cast<std::size_t>(j) * m + i] = v;
            }
        for (int j = 0; j < n; ++j)
            for (int a = 0; a < atoms; ++a) {
                double v = 0.;
                for (int i = 0; i < m; ++i) v += P[static_cast<std::size_t>(a) * m + i] * R[static_cast<std::size_t>(j) * m + i];
                G[static_cast<std::size_t>(j) * atoms + a] = Z[static_cast<std::size_t>(j) * atoms + a] + mu * v;
            }
        std::copy(X.begin(), X.end(), Xp.begin());
        for (int a = 0; a < atoms; ++a) {
            double norm2 = 0.;
            for (int j = 0; j < n; ++j) norm2 += G[static_cast<std::size_t>(j) * atoms + a] * G[static_cast<std::size_t>(j) * atoms + a];
            const double norm = std::sqrt(norm2), threshold = lam * weight[a];
            const double scale = norm <= threshold ? 0. : 1. - threshold / norm;
            for (int j = 0; j < n; ++j) X[static_cast<std::size_t>(j) * atoms + a] = G[static_cast<std::size_t>(j) * atoms + a] * scale;
        }
        const double next = (1. + std::sqrt(1. + 4. * t * t)) / 2., beta = (t - 1.) / next;
        t = next;
        for (std::size_t e = 0; e < X.size(); ++e) Z[e] = X[e] + beta * (X[e] - Xp[e]);
    }
    out.assign(static_cast<std::size_t>(m) * n, 0.);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            double v = mean[j];
            for (int a = 0; a < atoms; ++a) v += d(a, i) * X[static_cast<std::size_t>(j) * atoms + a];
            out[static_cast<std::size_t>(j) * m + i] = v;
        }
}

// The cluster coding against the FP64 model (fixed_group tolerance of
// contracts/tolerances.json), and at a vanishing sigma the reconstruction of
// a group from an overcomplete learned dictionary: the 16 ISTA rounds of model
// version 2 left this text-like group 0.054 (RMS) away; the preconditioned FISTA
// comes within 0.0051, under the 0.01 bound.
static int test_coding_reference() {
    constexpr int block = 8, m = block * block, w = 160, h = 160;
    std::vector<float> image(static_cast<std::size_t>(w) * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float v = 0.15f + 0.05f * std::sin(x / 7.f) * std::cos(y / 5.f) + 0.02f * std::sin((x + 2 * y) / 3.f);
            if ((x % 23 < 2 && y % 29 < 20) || (y % 29 < 2 && x % 23 < 15)) v = 0.9f;
            image[static_cast<std::size_t>(y) * w + x] = v;
        }
    const int nx = w - block + 1, total = nx * (h - block + 1);
    constexpr int pool = 600, atoms = 256, n = 48;
    std::vector<float> patches(static_cast<std::size_t>(pool) * m);
    for (int j = 0; j < pool; ++j) {
        const int at = static_cast<int>((static_cast<long long>(j) * 7919) % total), px = at % nx, py = at / nx;
        for (int i = 0; i < m; ++i)
            patches[static_cast<std::size_t>(j) * m + i] = image[static_cast<std::size_t>(py + i / block) * w + px + i % block];
    }
    std::vector<float> D(static_cast<std::size_t>(m) * atoms, 0.f);
    nss::lssc_dict_init(D.data(), m, atoms, m, patches.data(), pool, m, block, 1, 0x4C535343u);
    std::vector<float> prep(static_cast<std::size_t>(nss::lssc_prepare_work_floats(m, atoms)));
    nss::LsscPreparedContext context;
    if (nss::lssc_prepare_context(D.data(), m, atoms, m, prep.data(), static_cast<int>(prep.size()), &context) != 0 ||
        !context.lambda_scale || context.gradient == context.transpose) {
        return fail("lssc coding: no preconditioner for a learned dictionary");
    }
    std::vector<float> group(patches.begin(), patches.begin() + static_cast<std::ptrdiff_t>(n) * m);
    std::vector<float> work(static_cast<std::size_t>(nss::lssc_reconstruct_prepared_work_floats(m, n, atoms)));
    for (const double sigma : {1e-7, 3. / 255., 25. / 255.}) {
        double worst = 0.;
        std::vector<float> got = group;
        nss::lssc_reconstruct_prepared(got.data(), m, n, m, &context, static_cast<float>(sigma), work.data(),
                                       static_cast<int>(work.size()));
        std::vector<double> want;
        coding_reference(group, m, n, D, atoms, context.lipschitz, sigma, want);
        double rms = 0.;
        for (std::size_t e = 0; e < got.size(); ++e) {
            const double err = std::fabs(got[e] - want[e]);
            worst = std::max(worst, err);
            if (!(err <= 2e-5 + 2e-4 * std::fabs(want[e]))) {
                std::fprintf(stderr, "sigma %.3g element %zu: %.9g vs FP64 %.9g\n", sigma, e, got[e], want[e]);
                return fail("lssc coding differs from the FP64 model");
            }
            rms += (got[e] - group[e]) * (got[e] - group[e]);
        }
        rms = std::sqrt(rms / static_cast<double>(got.size()));
        std::printf("lssc coding sigma=%.3g: max |cpu - fp64| %.3g, rms change %.5f\n", sigma, worst, rms);
        if (sigma < 1e-6 && !(rms < 0.01)) {
            return fail("lssc coding at a vanishing sigma does not reconstruct the group");
        }
    }
    return 0;
}

static int test_denoise_psnr() {
    constexpr int w = 32;
    constexpr int h = 32;
    constexpr int block = 8;
    constexpr int step = 4;
    constexpr float noise = 0.08f;
    std::vector<float> clean(static_cast<std::size_t>(w * h));
    std::vector<float> noisy(static_cast<std::size_t>(w * h));
    std::vector<float> num(static_cast<std::size_t>(w * h), 0.f);
    std::vector<float> den(static_cast<std::size_t>(w * h), 0.f);
    std::vector<float> out(static_cast<std::size_t>(w * h), 0.f);
    std::mt19937 rng(42);
    std::normal_distribution<float> gauss(0.f, noise);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float v = 0.35f + 0.2f * std::sin(static_cast<float>(x) * 0.2f) +
                            0.2f * std::sin(static_cast<float>(y) * 0.15f);
            clean[static_cast<std::size_t>(y * w + x)] = v;
            noisy[static_cast<std::size_t>(y * w + x)] = v + gauss(rng);
        }
    }
    nss::lssc_denoise_plane(noisy.data(), w, h, w, num.data(), den.data(), w, block, step, noise);
    const int work_need = nss::lssc_denoise_work_floats(w, h, block, step);
    std::vector<float> explicit_work(static_cast<std::size_t>(work_need) + 8, -12345.5f);
    std::vector<float> num_ws(static_cast<std::size_t>(w * h), 0.f);
    std::vector<float> den_ws(static_cast<std::size_t>(w * h), 0.f);
    nss::lssc_denoise_plane(noisy.data(), w, h, w, num_ws.data(), den_ws.data(), w, block, step, noise,
                            explicit_work.data() + 4, work_need);
    for (int i = 0; i < 4; ++i) {
        if (explicit_work[static_cast<std::size_t>(i)] != -12345.5f ||
            explicit_work[static_cast<std::size_t>(work_need + 4 + i)] != -12345.5f) {
            return fail("denoise workspace guard was overwritten");
        }
    }
    if (!std::equal(num.begin(), num.end(), num_ws.begin(), [](float a, float b) { return std::fabs(a - b) <= 3e-5f; }) ||
        !std::equal(den.begin(), den.end(), den_ws.begin(), [](float a, float b) { return std::fabs(a - b) <= 3e-5f; })) {
        return fail("explicit denoise workspace differs from fallback");
    }
    nss::aggregate_finish(out.data(), num.data(), den.data(), noisy.data(), w, h, w, w);
    for (int i = 0; i < w * h; ++i) {
        if (!std::isfinite(out[static_cast<std::size_t>(i)])) {
            return fail("denoise plane non-finite");
        }
    }
    const double mse_n = mse(noisy.data(), clean.data(), w * h);
    const double mse_d = mse(out.data(), clean.data(), w * h);
    const double psnr_n = 10.0 * std::log10(1.0 / (mse_n + 1e-18));
    const double psnr_d = 10.0 * std::log10(1.0 / (mse_d + 1e-18));
    std::printf("lssc psnr_noisy=%.3f psnr_den=%.3f\n", psnr_n, psnr_d);
    if (!(psnr_d > psnr_n + 0.2)) {
        return fail("PSNR vs clean must beat noisy by > 0.2 dB");
    }
    return 0;
}

int main() {
    if (test_cluster_assigns_every_patch() != 0) {
        return 1;
    }
    if (test_reconstruct_finite() != 0) {
        return 1;
    }
    if (test_coding_reference() != 0) {
        return 1;
    }
    if (test_denoise_psnr() != 0) {
        return 1;
    }
    {
        constexpr int w = 128;
        constexpr int h = 128;
        constexpr int block = 8;
        constexpr int step = 8;
        constexpr float sigma = 25.f / 255.f;
        std::vector<float> clean(static_cast<std::size_t>(w * h));
        std::vector<float> noisy(static_cast<std::size_t>(w * h));
        std::vector<float> num(static_cast<std::size_t>(w * h), 0.f);
        std::vector<float> den(static_cast<std::size_t>(w * h), 0.f);
        std::vector<float> out(static_cast<std::size_t>(w * h), 0.f);
        std::mt19937 rng(1);
        std::normal_distribution<float> gauss(0.f, sigma);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const float v = 0.35f + 0.25f * std::sin(static_cast<float>(x) / 12.f) *
                                            std::cos(static_cast<float>(y) / 9.f);
                clean[static_cast<std::size_t>(y * w + x)] = v;
                noisy[static_cast<std::size_t>(y * w + x)] = v + gauss(rng);
            }
        }
        nss::lssc_denoise_plane(noisy.data(), w, h, w, num.data(), den.data(), w, block, step, sigma);
        nss::aggregate_finish(out.data(), num.data(), den.data(), noisy.data(), w, h, w, w);
        double mn = 1e30;
        double mx = -1e30;
        for (int i = 0; i < w * h; ++i) {
            const float v = out[static_cast<std::size_t>(i)];
            if (!std::isfinite(v) || std::fabs(v) > 8.f) {
                return fail("lssc 128 coding exploded");
            }
            mn = std::min(mn, static_cast<double>(v));
            mx = std::max(mx, static_cast<double>(v));
        }
        const double psnr_n = 10.0 * std::log10(1.0 / (mse(noisy.data(), clean.data(), w * h) + 1e-18));
        const double psnr_d = 10.0 * std::log10(1.0 / (mse(out.data(), clean.data(), w * h) + 1e-18));
        std::printf("lssc 128 psnr_noisy=%.3f psnr_den=%.3f range=[%.3f,%.3f]\n", psnr_n, psnr_d, mn, mx);
        if (!(psnr_d > psnr_n + 0.2)) {
            return fail("lssc 128 must beat noisy");
        }
    }
    std::printf("test_lssc ok\n");
    return 0;
}
