#include "nss/cpu_api.hpp"
#include "nss/cpu_batch.hpp"
#include "cpu/wnnm/jacobi8.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {
constexpr float sentinel = -12345.f;

bool gemm_empty_product() {
    for (int m : {1, 4, 8, 17, 64, 256}) {
        for (int n : {1, 4, 8, 9, 32}) {
            for (int pad : {0, 3}) {
                const int ldc = m + pad;
                std::vector<float> c(ldc * n + 1, sentinel);
                // An empty product must not access A/B, even for pointer arithmetic.
                nss::gemm_nn_hwy(m, n, 0, nullptr, m, nullptr, 0, c.data(), ldc);
                for (int j = 0; j < n; ++j) {
                    for (int i = 0; i < ldc; ++i) {
                        if (c[i + j * ldc] != (i < m ? 0.f : sentinel)) {
                            std::fprintf(stderr, "empty GEMM m=%d n=%d pad=%d\n", m, n, pad);
                            return false;
                        }
                    }
                }
                if (c.back() != sentinel) return false;
            }
        }
    }
    return true;
}

bool shrink_case(int m, int n, int count, int residual, int adaptive, float sigma, int pad_offset = 0) {
    std::vector<std::vector<float>> groups(count), work(count);
    std::vector<float> weights(count, -1.f);
    std::vector<int> status(count, 0);
    std::vector<nss::WnnmShrinkBatchItem> items(count);
    for (int b = 0; b < count; ++b) {
        const int lda = m + ((b + pad_offset) % 2 ? 3 : 0);
        groups[b].assign(lda * n + 1, sentinel);
        work[b].resize(nss::wnnm_shrink_work_floats(m, n));
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                groups[b][i + j * lda] = 0.25f + (j % 2 ? -1.f : 1.f) / 1024.f;
            }
        }
        items[b] = {groups[b].data(), m, n, lda, sigma, residual, adaptive, &weights[b],
                    work[b].data(), static_cast<int>(work[b].size()), &status[b]};
    }
    if (count == 1) {
        auto& a = items[0];
        if (nss::wnnm_shrink(a.group, m, n, a.lda, sigma, residual, adaptive, a.adaptive_weight,
                             a.work, a.work_floats) != 0) return false;
        status[0] = 1;
    } else if (nss::wnnm_shrink_batch(items.data(), count) != 0) {
        return false;
    }
    // Analytic rank-one fixture: residual=1,sigma=1 removes the entire
    // centered spectrum. residual=0 preserves S[0]; sigma=0 preserves input.
    const float mean = 0.25f + (n % 2 ? 1.f / (1024.f * n) : 0.f);
    for (int b = 0; b < count; ++b) {
        if (status[b] != 1 || weights[b] != 1.f) return false;
        const int lda = items[b].lda;
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < lda; ++i) {
                const float expected = i >= m ? sentinel :
                    (residual && sigma > 0.f ? mean : 0.25f + (j % 2 ? -1.f : 1.f) / 1024.f);
                const float actual = groups[b][i + j * lda];
                if (!std::isfinite(actual) || std::fabs(actual - expected) > 2e-6f) {
                    std::fprintf(stderr, "WNNM m=%d n=%d count=%d item=%d residual=%d sigma=%g: %g != %g\n",
                                 m, n, count, b, residual, sigma, actual, expected);
                    return false;
                }
            }
        }
        if (groups[b].back() != sentinel) return false;
    }
    return true;
}
}

int main() {
    bool ok = gemm_empty_product();
    for (int m : {1, 4, 8, 16, 64, 256}) {
        for (int n : {1, 2, 3, 4, 7, 8, 9, 16, 31, 32}) {
            for (int count : {1, 2, 5}) {
                for (int adaptive : {0, 1}) {
                    ok = shrink_case(m, n, count, 1, adaptive, 1.f) && ok;
                    if (count == 1) ok = shrink_case(m, n, count, 1, adaptive, 1.f, 1) && ok;
                }
            }
        }
    }
    for (int m : {4, 8, 64}) {
        for (int n : {4, 8}) {
            for (int count : {1, 5}) {
                ok = shrink_case(m, n, count, 0, 0, 1.f) && ok;
                ok = shrink_case(m, n, count, 1, 0, 0.f) && ok;
            }
        }
    }
    return ok ? 0 : 1;
}
