// SPDX-License-Identifier: GPL-2.0-only
#include "nss/ncsr_alignment_lab.hpp"
#include <cstdio>
int main() {
    if (!nss::alignment_lab::self_test()) return 1;
    std::puts("alignment eigensolver/weights/zero-sigma/noise-estimator checks passed");
    return 0;
}
