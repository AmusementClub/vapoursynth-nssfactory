#include "nss/cpu_api.hpp"
#include "nss/cpu_common.hpp"
#include "nss/cpu_lssc.hpp"
#include "cpu/wnnm/jacobi8.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace nss {
namespace {

float dot(const float* a, const float* b, int n) {
    return dot_n(a, b, n);
}

int chol_solve(int n, float* G, float* b) {
    if (n < 1) {
        return -1;
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            float sum = G[i * n + j];
            for (int p = 0; p < j; ++p) {
                sum -= G[i * n + p] * G[j * n + p];
            }
            if (i == j) {
                if (sum < 1e-12f) {
                    sum = 1e-12f;
                }
                G[i * n + i] = std::sqrt(sum);
            } else {
                G[i * n + j] = sum / G[j * n + j];
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        float s = b[i];
        for (int p = 0; p < i; ++p) {
            s -= G[i * n + p] * b[p];
        }
        b[i] = s / G[i * n + i];
    }
    for (int i = n - 1; i >= 0; --i) {
        float s = b[i];
        for (int p = i + 1; p < n; ++p) {
            s -= G[p * n + i] * b[p];
        }
        b[i] = s / G[i * n + i];
    }
    return 0;
}

}  // namespace

int lssc_omp(const float* y, int m, const float* D, int atoms, int ldd, int sparsity, float* a) {
    if (!y || !D || !a || m < 1 || atoms < 1 || ldd < m) {
        return 0;
    }
    std::memset(a, 0, static_cast<std::size_t>(atoms) * sizeof(float));
    int kmax = sparsity;
    if (kmax < 1) {
        return 0;
    }
    if (kmax > atoms) {
        kmax = atoms;
    }
    if (kmax > 8) {
        kmax = 8;
    }

    constexpr int kStackM = 256;
    constexpr int kStackA = 256;
    float r_s[kStackM];
    int supp_s[8];
    char used_s[kStackA];
    float Ds_s[kStackM * 8];
    float G_s[8 * 8];
    float b_s[8];
    std::vector<float> r_v;
    std::vector<int> supp_v;
    std::vector<char> used_v;
    std::vector<float> Ds_v;
    std::vector<float> G_v;
    std::vector<float> b_v;
    float* r = r_s;
    int* supp = supp_s;
    char* used = used_s;
    float* Ds = Ds_s;
    float* G = G_s;
    float* b = b_s;
    if (m > kStackM || atoms > kStackA) {
        r_v.resize(static_cast<std::size_t>(m));
        supp_v.assign(static_cast<std::size_t>(kmax), -1);
        used_v.assign(static_cast<std::size_t>(atoms), 0);
        Ds_v.assign(static_cast<std::size_t>(m) * static_cast<std::size_t>(kmax), 0.f);
        G_v.assign(static_cast<std::size_t>(kmax) * static_cast<std::size_t>(kmax), 0.f);
        b_v.assign(static_cast<std::size_t>(kmax), 0.f);
        r = r_v.data();
        supp = supp_v.data();
        used = used_v.data();
        Ds = Ds_v.data();
        G = G_v.data();
        b = b_v.data();
    } else {
        std::memset(used_s, 0, static_cast<std::size_t>(atoms));
        std::memset(Ds_s, 0, static_cast<std::size_t>(m) * static_cast<std::size_t>(kmax) * sizeof(float));
        std::memset(G_s, 0, static_cast<std::size_t>(kmax) * static_cast<std::size_t>(kmax) * sizeof(float));
        for (int i = 0; i < kmax; ++i) {
            supp_s[i] = -1;
        }
    }
    std::memcpy(r, y, static_cast<std::size_t>(m) * sizeof(float));
    float corr_s[kStackA];
    std::vector<float> corr_v;
    float* corr = corr_s;
    if (atoms > kStackA) {
        corr_v.assign(static_cast<std::size_t>(atoms), 0.f);
        corr = corr_v.data();
    }

    int ksel = 0;
    for (int t = 0; t < kmax; ++t) {
        gemm_tn_hwy(m, 1, atoms, D, ldd, r, m, corr, atoms);
        int best = -1;
        float best_abs = 0.f;
        for (int j = 0; j < atoms; ++j) {
            if (used[static_cast<std::size_t>(j)]) {
                continue;
            }
            const float ac = std::fabs(corr[static_cast<std::size_t>(j)]);
            if (ac > best_abs) {
                best_abs = ac;
                best = j;
            }
        }
        if (best < 0 || best_abs < 1e-12f) {
            break;
        }
        used[static_cast<std::size_t>(best)] = 1;
        supp[static_cast<std::size_t>(ksel)] = best;
        const float* db = D + static_cast<std::size_t>(best) * static_cast<std::size_t>(ldd);
        std::memcpy(Ds + static_cast<std::size_t>(ksel) * static_cast<std::size_t>(m), db,
                    static_cast<std::size_t>(m) * sizeof(float));
        ++ksel;

        for (int p = 0; p < ksel; ++p) {
            const float* dp = Ds + static_cast<std::size_t>(p) * static_cast<std::size_t>(m);
            for (int q = p; q < ksel; ++q) {
                const float* dq = Ds + static_cast<std::size_t>(q) * static_cast<std::size_t>(m);
                const float g = dot(dp, dq, m);
                G[p * ksel + q] = g;
                G[q * ksel + p] = g;
            }
            b[static_cast<std::size_t>(p)] = dot(dp, y, m);
        }
        if (chol_solve(ksel, G, b) != 0) {
            --ksel;
            used[static_cast<std::size_t>(best)] = 0;
            break;
        }
        std::memset(a, 0, static_cast<std::size_t>(atoms) * sizeof(float));
        for (int p = 0; p < ksel; ++p) {
            a[supp[static_cast<std::size_t>(p)]] = b[static_cast<std::size_t>(p)];
        }
        for (int i = 0; i < m; ++i) {
            float s = 0.f;
            for (int p = 0; p < ksel; ++p) {
                s += Ds[static_cast<std::size_t>(i) + static_cast<std::size_t>(p) * static_cast<std::size_t>(m)] *
                     b[static_cast<std::size_t>(p)];
            }
            r[static_cast<std::size_t>(i)] = y[i] - s;
        }
        float rn = 0.f;
        for (int i = 0; i < m; ++i) {
            rn += r[static_cast<std::size_t>(i)] * r[static_cast<std::size_t>(i)];
        }
        if (rn < 1e-12f) {
            break;
        }
    }
    return ksel;
}


void lssc_denoise_plane(const float* src, int width, int height, int sstride, float* num, float* den, int buf_stride,
                        int block, int step, float sigma, float* work, int work_floats) {
    if (!src || !num || !den || width < 1 || height < 1 || sstride < width || buf_stride < width) {
        return;
    }
    for (int y = 0; y < height; ++y) {
        std::memset(num + static_cast<std::size_t>(y) * static_cast<std::size_t>(buf_stride), 0,
                    static_cast<std::size_t>(width) * sizeof(float));
        std::memset(den + static_cast<std::size_t>(y) * static_cast<std::size_t>(buf_stride), 0,
                    static_cast<std::size_t>(width) * sizeof(float));
    }
    if (block < 1 || step < 1 || width < block || height < block) {
        return;
    }

    const int nx = lssc_axis_count(width, block, step);
    const int ny = lssc_axis_count(height, block, step);
    const int np = nx * ny;
    if (np < 1) {
        return;
    }
    const int m = block * block;
    const int lda = m;
    const int atoms = std::min(kLsscDefaultAtoms, np);
    const int nclusters = std::min(kLsscDefaultClusters, np);
    const int assign_f =
        (np * static_cast<int>(sizeof(int)) + static_cast<int>(sizeof(float)) - 1) / static_cast<int>(sizeof(float));
    const int counts_f = (nclusters * static_cast<int>(sizeof(int)) + static_cast<int>(sizeof(float)) - 1) /
                         static_cast<int>(sizeof(float));
    const int ista_n = lssc_reconstruct_work_floats(m, np, atoms);
    const int work_need = lssc_denoise_work_floats(width, height, block, step);

    std::vector<float> store;
    float* buf = work;
    if (!buf || work_floats < work_need) {
        store.assign(static_cast<std::size_t>(work_need), 0.f);
        buf = store.data();
    }
    float* patches = buf;
    float* D = patches + static_cast<std::size_t>(np) * static_cast<std::size_t>(lda);
    float* after_d = D + static_cast<std::size_t>(m) * static_cast<std::size_t>(atoms);
    int* assign = reinterpret_cast<int*>(after_d);
    float* after_assign = after_d + assign_f;
    int* counts = reinterpret_cast<int*>(after_assign);
    float* after_counts = after_assign + counts_f;
    int* members = reinterpret_cast<int*>(after_counts);
    float* group = after_counts + assign_f;
    float* ista = group + static_cast<std::size_t>(np) * static_cast<std::size_t>(m);
    std::memset(patches, 0, static_cast<std::size_t>(np) * static_cast<std::size_t>(lda) * sizeof(float));
    std::memset(D, 0, static_cast<std::size_t>(m) * static_cast<std::size_t>(atoms) * sizeof(float));

    std::vector<int> xs(static_cast<std::size_t>(np), 0);
    std::vector<int> ys(static_cast<std::size_t>(np), 0);
    int idx = 0;
    for (int by0 = 0; by0 < height - block + step; by0 += step) {
        const int by = std::min(by0, std::max(0, height - block));
        for (int bx0 = 0; bx0 < width - block + step; bx0 += step) {
            const int bx = std::min(bx0, std::max(0, width - block));
            pack_patch(patches + static_cast<std::size_t>(idx) * static_cast<std::size_t>(lda), lda, src, sstride, bx,
                       by, block, width, height);
            xs[static_cast<std::size_t>(idx)] = bx;
            ys[static_cast<std::size_t>(idx)] = by;
            ++idx;
        }
    }
    const int n = idx;
    if (n < 1) {
        return;
    }

    lssc_cluster(patches, m, n, lda, nclusters, assign, counts);
    lssc_dict_init(D, m, atoms, m, patches, n, lda, block, 1, 0x4C535343u);

    for (int c = 0; c < nclusters; ++c) {
        const int nc = counts[c];
        if (nc < 1) {
            continue;
        }
        int g = 0;
        for (int j = 0; j < n; ++j) {
            if (assign[j] != c) {
                continue;
            }
            std::memcpy(group + static_cast<std::size_t>(g) * static_cast<std::size_t>(m),
                        patches + static_cast<std::size_t>(j) * static_cast<std::size_t>(lda),
                        static_cast<std::size_t>(m) * sizeof(float));
            members[g] = j;
            ++g;
        }
        lssc_reconstruct(group, m, g, m, D, atoms, m, sigma, ista, ista_n);
        for (int t = 0; t < g; ++t) {
            const int j = members[t];
            std::memcpy(patches + static_cast<std::size_t>(j) * static_cast<std::size_t>(lda),
                        group + static_cast<std::size_t>(t) * static_cast<std::size_t>(m),
                        static_cast<std::size_t>(m) * sizeof(float));
        }
    }

    for (int j = 0; j < n; ++j) {
        const float* col = patches + static_cast<std::size_t>(j) * static_cast<std::size_t>(lda);
        for (int i = 0; i < m; ++i) {
            const float v = col[i];
            if (!std::isfinite(v) || std::fabs(v) > 8.f) {
                pack_patch(patches + static_cast<std::size_t>(j) * static_cast<std::size_t>(lda), lda, src, sstride,
                           xs[static_cast<std::size_t>(j)], ys[static_cast<std::size_t>(j)], block, width, height);
                break;
            }
        }
        aggregate_add(num, den, buf_stride, xs[static_cast<std::size_t>(j)], ys[static_cast<std::size_t>(j)], col, block,
                      width, height, 1.f);
    }
}

}  // namespace nss
