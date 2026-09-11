// Highway 1.4.0 has no ZA outer-product operation. Keep this ISA-specific leaf
// separate; packing, ordinary SIMD and fallback dispatch stay in Highway.
#include <arm_sme.h>
#include <arm_sve.h>
#include <cstdint>

extern "C" __arm_new("za") __arm_locally_streaming
void nss_lssc_sme_product(int m, int n, int k, const float* a, int lda,
                          const float* packed_b, float* c, int ldc) {
    const int lanes = static_cast<int>(svcntw());
    for (int j = 0; j < n; j += lanes) {
        const auto columns = svwhilelt_b32(j, n);
        for (int i = 0; i < m; i += lanes) {
            const auto rows = svwhilelt_b32(i, m);
            svzero_za();
            for (int t = 0; t < k; ++t) {
                const auto av = svld1_f32(rows, a + t * lda + i);
                const auto bv = svld1_f32(columns, packed_b + t * n + j);
                svmopa_za32_f32_m(0, rows, columns, av, bv);
            }
            const int count = n - j < lanes ? n - j : lanes;
            for (int column = 0; column < count; ++column)
                svst1_ver_za32(0, static_cast<std::uint32_t>(column), rows,
                              c + (j + column) * ldc + i);
        }
    }
}
