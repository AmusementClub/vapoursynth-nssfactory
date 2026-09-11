#include "nss/cpu_lssc.hpp"
#include "cpu/lssc/mixed_omp.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace nss {
int lssc_omp_workspace_reference(const float*, int, const float*, int, int, int, float*, float*, int);
}
namespace {
unsigned cases = 0;
bool compare(const std::vector<float>& dictionary, const std::vector<float>& input,
             int atoms, int stride, int sparsity) {
    const int m = int(input.size()), need = nss::lssc_omp_work_floats(m, atoms, sparsity);
    constexpr float canary = 9876.5f;
    std::vector<float> left(atoms + 2, canary), right(atoms + 2, canary);
    std::vector<float> a(need + 2, canary), b(need + 2, canary);
    const int k0 = nss::lssc_omp_workspace_reference(input.data(), m, dictionary.data(), atoms, stride,
                                                    sparsity, left.data() + 1, a.data() + 1, need);
    const int k1 = nss::lssc_omp_workspace(input.data(), m, dictionary.data(), atoms, stride,
                                        sparsity, right.data() + 1, b.data() + 1, need);
    ++cases;
    if (k0 != k1 || std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) ||
        a.front() != canary || a.back() != canary || b.front() != canary || b.back() != canary ||
        left.front() != canary || left.back() != canary) {
        std::fprintf(stderr, "mixed mismatch: case=%u m=%d atoms=%d sparsity=%d counts=%d/%d\n",
                     cases, m, atoms, sparsity, k0, k1);
        return false;
    }
    return true;
}
}
int main() {
    std::mt19937 random(729114);
    auto dyadic = [&] { return float(int(random() % 257) - 128) / 128.f; };
    for (int m : {1, 3, 4, 15, 16, 17, 64, 65, 256, 257, 1024}) {
        for (int atoms : {1, 7, 8, 15, 16, 17, 25, 64, 255, 256, 257}) {
            const int stride = m + 3;
            for (int scale : {-20, 0, 20}) {
                std::vector<float> d(stride * atoms, 4567.f), y(m);
                for (int a = 0; a < atoms; ++a) for (int row = 0; row < m; ++row)
                    d[a * stride + row] = std::ldexp(dyadic(), scale);
                for (float& value : y) value = dyadic();
                for (int sparsity : {1, 3, 8}) if (!compare(d, y, atoms, stride, sparsity)) return 1;
            }
        }
    }
    constexpr int m = 64, atoms = 64, stride = 67;
    std::vector<float> d(stride * atoms, 0.f), y(m, 0.f);
    for (int a = 0; a < atoms; ++a) d[a * stride + a] = 1.f;
    for (float second : {1.f, std::nextafter(1.f, 0.f), std::nextafter(1.f, 2.f)}) {
        y[0] = 1.f; y[1] = second;
        if (!compare(d, y, atoms, stride, 1) || !compare(d, y, atoms, stride, 8)) return 2;
    }
    for (float value : {std::nextafter(1e-12f, 0.f), 1e-12f, std::nextafter(1e-12f, 1.f)}) {
        std::fill(y.begin(), y.end(), 0.f); y[0] = value;
        if (!compare(d, y, atoms, stride, 8)) return 3;
    }
    for (int exponent : {-140, -120, 100, 125}) {
        for (int a = 0; a < atoms; ++a) d[a * stride + a] = std::ldexp(1.f, exponent);
        std::fill(y.begin(), y.end(), 0.f); y[0] = std::ldexp(.75f, 100);
        if (!compare(d, y, atoms, stride, 8)) return 4;
    }
    // Exercise the interval filter itself: separated winner, exact/ULP tie,
    // dense ambiguity, threshold and nonfinite/overflow fallback.
    std::fill(d.begin(), d.end(), 0.f);
    for (int a = 0; a < atoms; ++a) d[a * stride + a] = 1.f;
    nss::detail::MixedOmpSearch search(d.data(), m, atoms, stride);
    std::vector<float> storage(2 * m + 1, 0.f);
    std::vector<unsigned char> used(atoms, 0);
    auto residual = [&](double first, double second) {
        for (int i = 0; i < m; ++i) { const double v = i == 0 ? first : i == 1 ? second : .001; std::memcpy(storage.data() + 1 + 2 * i, &v, sizeof(v)); }
    };
    int best = -1; double value = 0; nss::detail::MixedOmpStats stats;
    residual(1., .5);
    if (!search.select(storage.data() + 1, used.data(), best, value, &stats) || best != 0 || value != 1.) return 5;
    residual(1., std::nextafter(1., 2.));
    if (!search.select(storage.data() + 1, used.data(), best, value, &stats) || best != 1 || value != std::nextafter(1., 2.)) return 6;
    residual(1., 1.);
    if (!search.select(storage.data() + 1, used.data(), best, value, &stats) || best != 0) return 7;
    for (int a = 0; a < atoms; ++a) { std::fill_n(d.data() + a * stride, m, 0.f); d[a * stride] = 1.f; }
    nss::detail::MixedOmpSearch ambiguous(d.data(), m, atoms, stride);
    if (ambiguous.select(storage.data() + 1, used.data(), best, value)) return 8;
    residual(1e100, .5);
    if (search.select(storage.data() + 1, used.data(), best, value)) return 9;
    if (stats.accepted != 3 || stats.refined_columns < 5) return 10;
    std::printf("%u complete OMP cases byte-exact; separated/tied/dense/overflow interval branches exercised\n", cases);
    return 0;
}
