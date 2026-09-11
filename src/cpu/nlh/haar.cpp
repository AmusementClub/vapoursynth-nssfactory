#include "nss/cpu_nlh.hpp"
#include "nss/cpu_nlh_full.hpp"
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
    if (!in || !out || n < 1 || n > 64 || (n & (n - 1))) {
        return;
    }
    Haar1dN(in, out, n);
}

void IHaar1d(const float* in, float* out, int n) {
    if (!in || !out || n < 1 || n > 64 || (n & (n - 1))) {
        return;
    }
    IHaar1dN(in, out, n);
}

void NlhHaar2d(float* matrix, int q, int n, bool inverse) {
#if NSS_ALIGNMENT_GENERIC
    float line[64];
    auto rows = [&]() {
        for (int i = 0; i < q; ++i) {
            for (int j = 0; j < n; ++j) line[j] = matrix[i + j * q];
            if (inverse) IHaar1dN(line, line, n); else Haar1dN(line, line, n);
            for (int j = 0; j < n; ++j) matrix[i + j * q] = line[j];
        }
    };
    auto columns = [&]() {
        for (int j = 0; j < n; ++j) {
            if (inverse) IHaar1dN(matrix + j * q, matrix + j * q, q);
            else Haar1dN(matrix + j * q, matrix + j * q, q);
        }
    };
    if (inverse) { rows(); columns(); } else { columns(); rows(); }
#else
    if (q == 4 && n == 16) Haar2d_4x16(matrix, inverse);
    else Haar2dFast(matrix, q, n, inverse);
#endif
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(Haar1d);
HWY_EXPORT(IHaar1d);
HWY_EXPORT(NlhHaar2d);

void haar1d(const float* in, float* out, int n) {
    HWY_DYNAMIC_DISPATCH(Haar1d)(in, out, n);
}

void ihaar1d(const float* in, float* out, int n) {
    HWY_DYNAMIC_DISPATCH(IHaar1d)(in, out, n);
}

void nlh_haar2d(float* matrix, int q, int n, bool inverse) {
    HWY_DYNAMIC_DISPATCH(NlhHaar2d)(matrix, q, n, inverse);
}

}  // namespace nss
#endif
