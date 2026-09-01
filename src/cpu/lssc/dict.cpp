#include "nss/cpu_api.hpp"
#include "nss/cpu_lssc.hpp"
#include "cpu/wnnm/jacobi8.hpp"

#include <algorithm>
#include <array>
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
    std::array<std::pair<int, int>, kSvdMaxM> order{};
    int order_count = 0;
    for (int v = 0; v < block; ++v) {
        for (int u = 0; u < block; ++u) {
            if (order_count < static_cast<int>(order.size())) {
                order[static_cast<std::size_t>(order_count++)] = {u + v, u + v * block};
            }
        }
    }
    std::sort(order.begin(), order.begin() + order_count);
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

bool ksvd_lite_workspace(float* D, int m, int atoms, int ldd, const float* patches, int n, int lda, int iters,
                         float* work, int work_floats) {
    if (iters < 1 || n < 2 || atoms < 1) {
        return true;
    }
    if (!work || work_floats < lssc_dict_work_floats(m, atoms, n, iters)) {
        return false;
    }
    const int ns = std::min(n, 256);
    const int sparsity = 8;
    const int nsvd = std::min(ns, kSvdMaxN);
    float* Y = work;
    float* A = Y + static_cast<std::size_t>(m) * static_cast<std::size_t>(ns);
    float* aj = A + static_cast<std::size_t>(atoms) * static_cast<std::size_t>(ns);
    float* E = aj + atoms;
    float* U = E + static_cast<std::size_t>(m) * static_cast<std::size_t>(nsvd);
    float* S = U + static_cast<std::size_t>(m) * static_cast<std::size_t>(nsvd);
    float* Vt = S + nsvd;
    int* supp = reinterpret_cast<int*>(Vt + static_cast<std::size_t>(nsvd) * static_cast<std::size_t>(nsvd));
    float* R = reinterpret_cast<float*>(supp + nsvd);
    float* dold = R + static_cast<std::size_t>(m) * static_cast<std::size_t>(ns);
    float* aold = dold + m;
    const int omp_work_n = lssc_omp_work_floats(m, atoms, sparsity);
    float* omp_work = aold + nsvd;

    std::fill(Y, Y + static_cast<std::size_t>(m) * static_cast<std::size_t>(ns), 0.f);
    std::fill(A, A + static_cast<std::size_t>(atoms) * static_cast<std::size_t>(ns), 0.f);
    std::fill(R, R + static_cast<std::size_t>(m) * static_cast<std::size_t>(ns), 0.f);
    for (int j = 0; j < ns; ++j) {
        const int src = (j * n) / ns;
        const float* p = patches + static_cast<std::size_t>(src) * static_cast<std::size_t>(lda);
        float mean = 0.f;
        for (int i = 0; i < m; ++i) {
            mean += p[i];
        }
        mean /= static_cast<float>(m);
        float* yj = Y + static_cast<std::size_t>(j) * static_cast<std::size_t>(m);
        for (int i = 0; i < m; ++i) {
            yj[i] = p[i] - mean;
        }
    }

    std::fill(aj, aj + atoms, 0.f);
    std::fill(E, E + static_cast<std::size_t>(m) * static_cast<std::size_t>(nsvd), 0.f);
    std::fill(U, U + static_cast<std::size_t>(m) * static_cast<std::size_t>(nsvd), 0.f);
    std::fill(S, S + nsvd, 0.f);
    std::fill(Vt, Vt + static_cast<std::size_t>(nsvd) * static_cast<std::size_t>(nsvd), 0.f);
    std::fill(supp, supp + nsvd, 0);
    std::fill(dold, dold + m, 0.f);
    std::fill(aold, aold + nsvd, 0.f);

    for (int it = 0; it < iters; ++it) {
        for (int j = 0; j < ns; ++j) {
            lssc_omp_workspace(Y + static_cast<std::size_t>(j) * static_cast<std::size_t>(m), m, D, atoms, ldd,
                               sparsity, aj, omp_work, omp_work_n);
            for (int k = 0; k < atoms; ++k) {
                A[k + j * atoms] = aj[k];
            }
        }
        gemm_nn_hwy(m, ns, atoms, D, ldd, A, atoms, R, m);
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
            std::memcpy(dold, dk, static_cast<std::size_t>(m) * sizeof(float));
            for (int t = 0; t < nsup; ++t) {
                const int j = supp[static_cast<std::size_t>(t)];
                    aold[static_cast<std::size_t>(t)] = A[k + j * atoms];
                const float* yj = Y + static_cast<std::size_t>(j) * static_cast<std::size_t>(m);
                const float* rj = R + static_cast<std::size_t>(j) * static_cast<std::size_t>(m);
                float* et = E + static_cast<std::size_t>(t) * static_cast<std::size_t>(m);
                const float akj = aold[static_cast<std::size_t>(t)];
                for (int i = 0; i < m; ++i) {
                    et[i] = yj[i] - rj[i] + dold[static_cast<std::size_t>(i)] * akj;
                }
            }
            if (m > kSvdMaxM || nsup > kSvdMaxN) {
                continue;
            }
            if (svd_economy(m, nsup, E, m, U, m, S, Vt, nsup) != 0) {
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
                float* rj = R + static_cast<std::size_t>(j) * static_cast<std::size_t>(m);
                for (int i = 0; i < m; ++i) {
                    rj[i] += dk[i] * an - dold[static_cast<std::size_t>(i)] * ao;
                }
            }
        }
    }
    return true;
}

}  // namespace

void lssc_dict_init_workspace(float* D, int m, int atoms, int ldd, const float* patches, int n, int lda, int block,
                              int ksvd_iters, unsigned seed, float* work, int work_floats) {
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
    if (patches && n > 0 && lda >= m && iters > 0) {
        (void)ksvd_lite_workspace(D, m, atoms, ldd, patches, n, lda, iters, work, work_floats);
    }
}

void lssc_dict_init(float* D, int m, int atoms, int ldd, const float* patches, int n, int lda, int block,
                    int ksvd_iters, unsigned seed) {
    const int need = lssc_dict_work_floats(m, atoms, n, ksvd_iters);
    std::vector<float> work(static_cast<std::size_t>(std::max(1, need)), 0.f);
    lssc_dict_init_workspace(D, m, atoms, ldd, patches, n, lda, block, ksvd_iters, seed, work.data(), need);
}

}  // namespace nss
