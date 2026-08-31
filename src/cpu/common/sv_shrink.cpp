#include "nss/cpu_common.hpp"

#include <algorithm>
#include <cmath>

namespace nss {

int sv_shrink(float* S, int n, float constant, int start_k) {
    if (!S || n < 1) {
        return 0;
    }
    int k = std::clamp(start_k, 0, n);
    for (; k < n; ++k) {
        const float s = S[k];
        const float tmp = s * s - constant;
        if (tmp > 0.f) {
            S[k] = (s + std::sqrt(tmp)) * 0.5f;
        } else {
            break;
        }
    }
    return k;
}

}  // namespace nss
