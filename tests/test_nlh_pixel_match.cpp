#include "nss/backend.hpp"
#include "nss/cpu_nlh.hpp"
#include "hwy/highway.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <vector>

int main(int argc, char** argv) {
    if (argc > 1) {
        const auto target = std::string_view(argv[1]) == "avx2" ? HWY_AVX2 :
                            std::string_view(argv[1]) == "avx3" ? HWY_AVX3 : 0;
        const auto caps = nss::backend_caps();
        if (!target || !(caps.compiled_targets & caps.runtime_targets & target)) return 77;
        hwy::SetSupportedTargetsForTest(target);
    }
    std::ofstream output;
    if (argc > 2) {
        output.open(argv[2], std::ios::binary);
        if (!output) return 2;
    }
    auto emit = [&](int m, int n, int q, int pattern, const std::vector<int>& indices) {
        if (!output.is_open()) return;
        const std::int32_t header[]{m, n, q, pattern};
        output.write(reinterpret_cast<const char*>(header), sizeof(header));
        output.write(reinterpret_cast<const char*>(indices.data()),
                     std::streamsize(indices.size() * sizeof(int)));
    };
    try {
        for (int m : {4, 49, 64, 65, 81, 100, 256}) for (int n : {1, 16, 64})
        for (int pattern = 0; pattern < 3; ++pattern) {
            const int lda = m + 3, count = std::min(m, 16);
            std::vector<float> group(lda * n, std::numeric_limits<float>::quiet_NaN());
            for (int c = 0; c < n; ++c) for (int r = 0; r < m; ++r)
                group[r + c * lda] = pattern == 0 ? .375f :
                    pattern == 1 ? float(((r % 11) * 37 + c * 53) % 257 - 128) / 128 :
                    float(std::sin((r + c * 19) * .137) + .3 * std::cos((r * 7 + c) * .173));
            std::vector<int> expected(m * count);
            for (int r = 0; r < m; ++r) {
                std::vector<std::pair<float, int>> distances;
                for (int s = 0; s < m; ++s) if (s != r) {
                    float value = 0;
                    for (int c = 0; c < n; ++c) {
                        const float d = group[s + c * lda] - group[r + c * lda];
                        value = std::fma(d, d, value);
                    }
                    distances.emplace_back(value, s);
                }
                std::sort(distances.begin(), distances.end());
                expected[r * count] = r;
                for (int k = 1; k < count; ++k) expected[r * count + k] = distances[k - 1].second;
            }
            for (int q : {1, 2, 3, 4, 8, 16}) if (q <= m) {
                std::vector<int> indices(m * q, -1);
                nss::pixel_match(group.data(), m, n, lda, q, indices.data());
                for (int r = 0; r < m; ++r) for (int k = 0; k < q; ++k)
                    if (indices[r * q + k] != expected[r * count + k])
                        throw std::runtime_error("pixel selection differs from scalar FMA-distance ordering");
                emit(m, n, q, pattern, indices);
            }
        }
        // These outputs are compared with the frozen library separately. They
        // preserve its finite sentinel ties and overflow-distance behavior.
        if (output.is_open()) for (int m : {65, 81, 100, 256}) for (int n : {1, 16, 64})
        for (int pattern = 0; pattern < 3; ++pattern) for (int q : {2, 4, 8, 16}) {
            const int lda = m + 3;
            std::vector<float> group(lda * n, std::numeric_limits<float>::quiet_NaN());
            for (int c = 0; c < n; ++c) for (int r = 0; r < m; ++r)
                group[r + c * lda] = float(((r % 7) * 13 + c * 17) % 37 - 18) *
                                    (pattern == 0 ? 1.e14f : pattern == 1 ? 1.e19f : 1.e-20f);
            std::vector<int> indices(m * q);
            nss::pixel_match(group.data(), m, n, lda, q, indices.data());
            emit(m, n, q, 100 + pattern, indices);
        }
        if (output.is_open() && !output) return 2;
        std::cout << "NLH pixel rows: scalar FMA oracle, self-first order, ties and padded tails PASS\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
