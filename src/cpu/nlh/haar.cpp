#include "nss/cpu_nlh.hpp"
#include "cpu/hwy_config.hpp"

#include <cstring>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/nlh/haar.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

#include "cpu/nlh/haar_inl.hpp"

void Haar1d(const float* in, float* out, int n) {
    if (!in || !out || n < 1 || (n != 1 && n != 2 && n != 4 && n != 8 && n != 16)) {
        return;
    }
    Haar1dN(in, out, n);
}

void IHaar1d(const float* in, float* out, int n) {
    if (!in || !out || n < 1 || (n != 1 && n != 2 && n != 4 && n != 8 && n != 16)) {
        return;
    }
    IHaar1dN(in, out, n);
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(Haar1d);
HWY_EXPORT(IHaar1d);

void haar1d(const float* in, float* out, int n) {
    HWY_DYNAMIC_DISPATCH(Haar1d)(in, out, n);
}

void ihaar1d(const float* in, float* out, int n) {
    HWY_DYNAMIC_DISPATCH(IHaar1d)(in, out, n);
}

}  // namespace nss
#endif
