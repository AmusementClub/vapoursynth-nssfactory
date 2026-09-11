// Native Linux capability check. No unsupported ISA is executed.
#include <asm/hwcap.h>
#include <cerrno>
#include <cstdio>
#include <initializer_list>
#include <sys/auxv.h>
#include <sys/prctl.h>
#include <arm_sve.h>

__attribute__((target("arch=armv9-a+sve2"), noinline))
static unsigned sve_float_lanes() { return svcntw(); }

int main() {
    const auto hw = getauxval(AT_HWCAP), hw2 = getauxval(AT_HWCAP2);
    const bool sve = (hw & HWCAP_SVE) != 0, sve2 = (hw2 & HWCAP2_SVE2) != 0;
    const bool sme = (hw2 & HWCAP2_SME) != 0;
    errno = 0;
    const int sve_vl = prctl(PR_SVE_GET_VL), sve_errno = errno;
    errno = 0;
    const int sme_vl = prctl(PR_SME_GET_VL), sme_errno = errno;
    std::printf("{\"hwcap\":%lu,\"hwcap2\":%lu,\"sve\":%s,\"sve2\":%s,\"sme\":%s,"
                "\"sve_get_vl\":%d,\"sve_errno\":%d,\"sme_get_vl\":%d,\"sme_errno\":%d,"
                "\"float_lanes\":%u,\"sve_f32mm\":%s,\"sve_f64mm\":%s,\"sve_bf16\":%s,\"vl_requests\":[",
                hw, hw2, sve?"true":"false", sve2?"true":"false", sme?"true":"false",
                sve_vl, sve_errno, sme_vl, sme_errno, sve2?sve_float_lanes():0,
                (hw2&HWCAP2_SVEF32MM)?"true":"false",(hw2&HWCAP2_SVEF64MM)?"true":"false",
                (hw2&HWCAP2_SVEBF16)?"true":"false");
    if (sve) {
        bool first = true;
        for (int bytes : {16, 32, 64, 128, 256}) {
            errno = 0;
            const int actual = prctl(PR_SVE_SET_VL, bytes);
            std::printf("%s{\"requested_bytes\":%d,\"result\":%d,\"errno\":%d}",
                        first?"":",", bytes, actual, errno);
            first = false;
        }
        if (sve_vl >= 0) prctl(PR_SVE_SET_VL, sve_vl);
    }
    std::puts("]}");
}
