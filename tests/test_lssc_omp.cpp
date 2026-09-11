#include "nss/cpu_lssc.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
bool check(const std::vector<float>& y, const std::vector<float>& dictionary, int atoms, int stride,
           const std::vector<int>& expected_support, int sparsity = 8) {
    const int m = static_cast<int>(y.size());
    const int need = nss::lssc_omp_work_floats(m, atoms, sparsity);
    constexpr float sentinel = 12345.5f;
    // Offset by one float to exercise scratch without double alignment.
    std::vector<float> storage(need + 2, sentinel), result(atoms + 2, sentinel), wrapper(atoms);
    const int count = nss::lssc_omp_workspace(y.data(), m, dictionary.data(), atoms, stride, sparsity,
                                             result.data() + 1, storage.data() + 1, need);
    const int other = nss::lssc_omp(y.data(), m, dictionary.data(), atoms, stride, sparsity, wrapper.data());
    if (storage.front() != sentinel || storage.back() != sentinel || result.front() != sentinel ||
        result.back() != sentinel || count != other || count != int(expected_support.size()) ||
        !std::equal(wrapper.begin(), wrapper.end(), result.begin() + 1)) {
        std::fprintf(stderr, "OMP count/workspace mismatch: m=%d atoms=%d count=%d expected=%zu\n",
                     m, atoms, count, expected_support.size());
        return false;
    }
    for (int atom = 0; atom < atoms; ++atom) {
        const bool expected = std::find(expected_support.begin(), expected_support.end(), atom) != expected_support.end();
        if (!std::isfinite(wrapper[atom]) || (wrapper[atom] != 0.f) != expected) {
            std::fprintf(stderr, "OMP spurious/missing support: m=%d atom=%d coefficient=%.9g\n", m, atom, wrapper[atom]);
            return false;
        }
    }
    // Independent modified Gram-Schmidt QR in long double. This also accounts
    // for the small loss of orthogonality in the stored FP32 DCT dictionary.
    const int rank = static_cast<int>(expected_support.size());
    std::vector<long double> q(m * rank), r(rank * rank), expected(rank);
    for (int col = 0; col < rank; ++col) {
        for (int row = 0; row < m; ++row) q[row + col * m] = dictionary[row + expected_support[col] * stride];
        for (int p = 0; p < col; ++p) {
            long double dot = 0;
            for (int row = 0; row < m; ++row) dot += q[row + p * m] * q[row + col * m];
            r[p * rank + col] = dot;
            for (int row = 0; row < m; ++row) q[row + col * m] -= dot * q[row + p * m];
        }
        long double norm = 0;
        for (int row = 0; row < m; ++row) norm += q[row + col * m] * q[row + col * m];
        r[col * rank + col] = std::sqrt(norm);
        for (int row = 0; row < m; ++row) {
            q[row + col * m] /= r[col * rank + col];
            expected[col] += q[row + col * m] * y[row];
        }
    }
    for (int col = rank - 1; col >= 0; --col) {
        for (int p = col + 1; p < rank; ++p) expected[col] -= r[col * rank + p] * expected[p];
        expected[col] /= r[col * rank + col];
    }
    for (int col = 0; col < rank; ++col) {
        const int atom = expected_support[col];
        if (std::abs(static_cast<long double>(wrapper[atom]) - expected[col]) > 2e-7L * std::max(1.L, std::abs(expected[col]))) {
            std::fprintf(stderr, "OMP projection error: m=%d atom=%d\n", m, atom);
            return false;
        }
    }
    return true;
}
}

int main() {
    // A linear 8x8 patch has energy only in odd horizontal/vertical DCT atoms.
    // In the first 25 zigzag atoms these are 1,2,6,9,15,20. Its residual still
    // contains omitted higher frequencies: residual norm alone cannot stop
    // OMP, and FP32 cancellation previously selected numerical null atoms.
    constexpr int m = 64, atoms = 25, stride = 69;
    std::vector<float> dictionary(atoms * stride, 777.f), y(m);
    const int dw_size = nss::lssc_dict_work_floats(m, atoms, 1, 0);
    std::vector<float> dw(dw_size);
    nss::lssc_dict_init_workspace(dictionary.data(), m, atoms, stride, nullptr, 0, m, 8, 0, 1,
                                   dw.data(), dw_size, false);
    for (int exponent : {-8, -4, 0, 4, 8}) {
        for (int row = 0; row < 8; ++row) for (int col = 0; col < 8; ++col)
            y[row * 8 + col] = std::ldexp(float(3 * col + 5 * row - 28) / 128.f, exponent);
        if (!check(y, dictionary, atoms, stride, {1, 2, 6, 9, 15, 20})) return 1;
    }
    // Exact one-hot dictionaries cover the caller-sized residual scratch,
    // padded dictionary strides, sparsity=1 and the >256-row wrapper path.
    for (int rows : {1, 3, 4, 8, 17, 64, 65, 256, 257, 1024}) {
        const int n = std::min(9, rows), ld = rows + 3;
        std::vector<float> d(n * ld, 0.f), input(rows, 0.f);
        for (int i = 0; i < n; ++i) d[i * ld + i] = 1.f;
        input[0] = .5f;
        if (!check(input, d, n, ld, {0}, 1)) return 2;
        if (n >= 3) {
            input[2] = -.25f;
            if (!check(input, d, n, ld, {0, 2})) return 3;
        }
    }
    std::puts("LSSC OMP projection, numerical-null support and exact workspace tests passed");
    return 0;
}
