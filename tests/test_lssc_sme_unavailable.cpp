#include "nss/cpu_lssc.hpp"
#include "cpu/lssc/gemm.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>
#if defined(__APPLE__)
#include <dlfcn.h>
#include <sys/sysctl.h>
namespace { int sme_queries = 0; }
// Link-time test fixture: the real SME object remains in the binary, while
// the OS capability query reports an unsupported CPU. Production has no hook.
extern "C" int sysctlbyname(const char* name, void* value, size_t* size,
                            void* replacement, size_t replacement_size) {
    if (std::strcmp(name, "hw.optional.arm.FEAT_SME") == 0) {
        ++sme_queries;
        if (!value || !size || *size < sizeof(int)) return -1;
        *static_cast<int*>(value) = 0;
        *size = sizeof(int);
        return 0;
    }
    static const auto real = reinterpret_cast<decltype(&sysctlbyname)>(dlsym(RTLD_NEXT, "sysctlbyname"));
    return real ? real(name, value, size, replacement, replacement_size) : -1;
}
#endif

int main() {
    if (nss::lssc_sme_compiled() != bool(NSS_EXPECT_SME_COMPILED)) return 1;
    if (nss::lssc_sme_available() || nss::lssc_gemm_pack_work_floats(64, 129, 64)) return 2;
    std::vector<float> a(64 * 64, 1.f), b(64 * 129, .5f), c(64 * 129, -1.f);
    if (nss::lssc_gemm_nn(64, 129, 64, a.data(), 64, b.data(), 64, c.data(), 64, nullptr, 0)) return 3;
    if (!std::all_of(c.begin(), c.end(), [](float value) { return value == 32.f; })) return 4;
#if defined(__APPLE__)
    if (NSS_EXPECT_SME_COMPILED && sme_queries != 1) return 5;
#endif
    std::printf("SME_compiled=%d SME_available=0 fallback_product_exact=1\n", NSS_EXPECT_SME_COMPILED);
    return 0;
}
