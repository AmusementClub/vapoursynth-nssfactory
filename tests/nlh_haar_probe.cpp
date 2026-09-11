// Binary evidence for same-target Haar operation ordering across every shape.
#include "nss/cpu_nlh_full.hpp"
#include "nss/backend.hpp"
#include "hwy/highway.h"
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string_view>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) return 2;
    if (argc == 3) {
        const auto target = std::string_view(argv[2]) == "avx2" ? HWY_AVX2 :
                            std::string_view(argv[2]) == "avx3" ? HWY_AVX3 : 0;
        const auto caps = nss::backend_caps();
        if (!target || !(caps.compiled_targets & caps.runtime_targets & target)) return 77;
        hwy::SetSupportedTargetsForTest(target);
    }
    std::ofstream output(argv[1], std::ios::binary);
    if (!output) return 2;
    for (int q : {1, 2, 4, 8, 16}) for (int n : {1, 2, 4, 8, 16, 32, 64}) {
        for (int pattern = 0; pattern < 4; ++pattern) {
            const std::int32_t header[]{q, n, pattern};
            output.write(reinterpret_cast<const char*>(header), sizeof(header));
            std::vector<float> matrix(q * n);
            for (int i = 0; i < q * n; ++i) {
                matrix[i] = pattern == 0 ? float((i * 37 + 19) % 257 - 128) / 128 :
                            pattern == 1 ? float(std::sin(i * .131) + .3 * std::cos(i * .047)) :
                            pattern == 2 ? .375f : ((i % 3) ? 1.e-20f : -1.e-20f);
            }
            output.write(reinterpret_cast<const char*>(matrix.data()), std::streamsize(matrix.size() * sizeof(float)));
            nss::nlh_haar2d(matrix.data(), q, n, false);
            output.write(reinterpret_cast<const char*>(matrix.data()), std::streamsize(matrix.size() * sizeof(float)));
            nss::nlh_haar2d(matrix.data(), q, n, true);
            output.write(reinterpret_cast<const char*>(matrix.data()), std::streamsize(matrix.size() * sizeof(float)));
        }
    }
    return output ? 0 : 1;
}
