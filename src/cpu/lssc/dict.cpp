#include "nss/cpu_api.hpp"
#include "nss/cpu_lssc.hpp"
#include "cpu/wnnm/jacobi8.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace nss {
namespace {

void normalize_atom(float* d, int m) {
    float n2 = 0.f;
    for (int i = 0; i < m; ++i) {
        n2 += d[i] * d[i];
    }
    if (n2 < 1e-12f) {
        std::memset(d, 0, static_cast<std::size_t>(m) * sizeof(float));
        d[0] = 1.f;
        return;
    }
    const float inv = 1.f / std::sqrt(n2);
    for (int i = 0; i < m; ++i) {
        d[i] *= inv;
    }
}

unsigned lcg(unsigned& s) {
    s = s * 1664525u + 1013904223u;
    return s;
}

void fill_dct_atoms(float* D, int m, int atoms, int ldd, int block) {
    if (block < 1 || block * block != m) {
        return;
    }
    const int n_dct = std::min(atoms, m);
    std::vector<std::pair<int, int>> order;
    order.reserve(static_cast<std::size_t>(m));
    for (int v = 0; v < block; ++v) {
        for (int u = 0; u < block; ++u) {
            order.push_back({u + v, u + v * block});
        }
    }
    std::sort(order.begin(), order.end());
    for (int a = 0; a < n_dct; ++a) {
        float* col = D + static_cast<std::size_t>(a) * static_cast<std::size_t>(ldd);
        std::memset(col, 0, static_cast<std::size_t>(m) * sizeof(float));
        col[order[static_cast<std::size_t>(a)].second] = 1.f;
        idct_2d(col, block);
        normalize_atom(col, m);
    }
}

void fill_patch_atoms(float* D, int m, int atoms, int ldd, const float* patches, int n, int lda, int start,
                      unsigned seed) {
    if (start >= atoms || n < 1) {
        return;
    }
    unsigned s = seed ? seed : 1u;
    for (int a = start; a < atoms; ++a) {
        float* col = D + static_cast<std::size_t>(a) * static_cast<std::size_t>(ldd);
        const int src = static_cast<int>(lcg(s) % static_cast<unsigned>(n));
        const float* p = patches + static_cast<std::size_t>(src) * static_cast<std::size_t>(lda);
        float mean = 0.f;
        for (int i = 0; i < m; ++i) {
            mean += p[i];
        }
        mean /= static_cast<float>(m);
        for (int i = 0; i < m; ++i) {
            col[i] = p[i] - mean;
        }
        normalize_atom(col, m);
    }
}

void ksvd_lite(float* D, int m, int atoms, int ldd, const float* patches, int n, int lda, int iters) {
    if (iters < 1 || n < 2 || atoms < 1) {
        return;
    }
    const int ns = std::min(n, 256);
    const int sparsity = 8;
    std::vector<float> Y(static_cast<std::size_t>(m) * static_cast<std::size_t>(ns), 0.f);
    for (int j = 0; j < ns; ++j) {
        const int src = (j * n) / ns;
        const float* p = patches + static_cast<std::size_t>(src) * static_cast<std::size_t>(lda);
        float mean = 0.f;
        for (int i = 0; i < m; ++i) {
            mean += p[i];
        }
        mean /= static_cast<float>(m);
        float* yj = Y.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(m);
        for (int i = 0; i < m; ++i) {
            yj[i] = p[i] - mean;
        }
    }

    std::vector<float> A(static_cast<std::size_t>(atoms) * static_cast<std::size_t>(ns), 0.f);
    std::vector<float> aj(static_cast<std::size_t>(atoms), 0.f);
    const int nsvd = std::min(ns, kSvdMaxN);
    std::vector<float> E(static_cast<std::size_t>(m) * static_cast<std::size_t>(nsvd), 0.f);
    std::vector<float> U(static_cast<std::size_t>(m) * static_cast<std::size_t>(nsvd), 0.f);
    std::vector<float> S(static_cast<std::size_t>(nsvd), 0.f);
    std::vector<float> Vt(static_cast<std::size_t>(nsvd) * static_cast<std::size_t>(nsvd), 0.f);
    std::vector<int> supp(static_cast<std::size_t>(nsvd), 0);
    std::vector<float> R(static_cast<std::size_t>(m) * static_cast<std::size_t>(ns), 0.f);
    std::vector<float> dold(static_cast<std::size_t>(m), 0.f);
    std::vector<float> aold(static_cast<std::size_t>(nsvd), 0.f);

    for (int it = 0; it < iters; ++it) {
        for (int j = 0; j < ns; ++j) {
            lssc_omp(Y.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(m), m, D, atoms, ldd, sparsity,
                     aj.data());
            for (int k = 0; k < atoms; ++k) {
                A[k + j * atoms] = aj[static_cast<std::size_t>(k)];
            }
        }
        gemm_nn_hwy(m, ns, atoms, D, ldd, A.data(), atoms, R.data(), m);
        for (int k = 0; k < atoms; ++k) {
            int nsup = 0;
            for (int j = 0; j < ns && nsup < nsvd; ++j) {
                if (std::fabs(A[k + j * atoms]) > 1e-8f) {
                    supp[static_cast<std::size_t>(nsup++)] = j;
                }
            }
            if (nsup < 1) {
                continue;
            }
            float* dk = D + static_cast<std::size_t>(k) * static_cast<std::size_t>(ldd);
            std::memcpy(dold.data(), dk, static_cast<std::size_t>(m) * sizeof(float));
            for (int t = 0; t < nsup; ++t) {
                const int j = supp[static_cast<std::size_t>(t)];
                aold[static_cast<std::size_t>(t)] = A[k + j * atoms];
                const float* yj = Y.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(m);
                const float* rj = R.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(m);
                float* et = E.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(m);
                const float akj = aold[static_cast<std::size_t>(t)];
                for (int i = 0; i < m; ++i) {
                    et[i] = yj[i] - rj[i] + dold[static_cast<std::size_t>(i)] * akj;
                }
            }
            if (m > kSvdMaxM || nsup > kSvdMaxN) {
                continue;
            }
            if (svd_economy(m, nsup, E.data(), m, U.data(), m, S.data(), Vt.data(), nsup) != 0) {
                continue;
            }
            float sign = 1.f;
            float ip = 0.f;
            for (int i = 0; i < m; ++i) {
                ip += dold[static_cast<std::size_t>(i)] * U[static_cast<std::size_t>(i)];
            }
            if (ip < 0.f) {
                sign = -1.f;
            }
            for (int i = 0; i < m; ++i) {
                dk[i] = sign * U[static_cast<std::size_t>(i)];
            }
            normalize_atom(dk, m);
            const float s0 = S[0];
            for (int t = 0; t < nsup; ++t) {
                const int j = supp[static_cast<std::size_t>(t)];
                const float an = sign * s0 * Vt[static_cast<std::size_t>(t) * static_cast<std::size_t>(nsup)];
                const float ao = aold[static_cast<std::size_t>(t)];
                A[k + j * atoms] = an;
                float* rj = R.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(m);
                for (int i = 0; i < m; ++i) {
                    rj[i] += dk[i] * an - dold[static_cast<std::size_t>(i)] * ao;
                }
            }
        }
    }
}

}  // namespace

void lssc_dict_init(float* D, int m, int atoms, int ldd, const float* patches, int n, int lda, int block,
                    int ksvd_iters, unsigned seed) {
    if (!D || m < 1 || atoms < 1 || ldd < m) {
        return;
    }
    std::memset(D, 0, static_cast<std::size_t>(atoms) * static_cast<std::size_t>(ldd) * sizeof(float));
    fill_dct_atoms(D, m, atoms, ldd, block);
    const int n_dct = (block > 0 && block * block == m) ? std::min(atoms, m) : 0;
    if (n_dct < atoms) {
        if (patches && n > 0 && lda >= m) {
            fill_patch_atoms(D, m, atoms, ldd, patches, n, lda, n_dct, seed);
        } else {
            for (int a = n_dct; a < atoms; ++a) {
                float* col = D + static_cast<std::size_t>(a) * static_cast<std::size_t>(ldd);
                col[a % m] = 1.f;
            }
        }
    }
    for (int a = 0; a < atoms; ++a) {
        normalize_atom(D + static_cast<std::size_t>(a) * static_cast<std::size_t>(ldd), m);
    }
    int iters = ksvd_iters;
    if (iters < 0) {
        iters = 0;
    }
    if (iters > 8) {
        iters = 8;
    }
    if (patches && n > 0 && lda >= m) {
        ksvd_lite(D, m, atoms, ldd, patches, n, lda, iters);
    }
}

}  // namespace nss
