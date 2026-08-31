#include "nss/cpu_api.hpp"
#include "nss/cpu_common.hpp"

namespace nss {

void pack_patch_nch(float* col, int lda, const float* const* srcs, const int* strides, int nch, int x, int y,
                    int block, int width, int height) {
    // lda is the caller column stride (patches + i*lda). Channel packing inside a column is dense p².
    (void)lda;
    if (!col || !srcs || !strides || nch < 1 || block < 1) {
        return;
    }
    const int area = block * block;
    for (int c = 0; c < nch; ++c) {
        if (!srcs[c]) {
            continue;
        }
        pack_patch(col + c * area, area, srcs[c], strides[c], x, y, block, width, height);
    }
}

void unpack_patch_nch(float* const* nums, float* const* dens, const int* strides, int nch, int x, int y,
                      const float* col, int block, int width, int height, float w) {
    if (!nums || !dens || !strides || !col || nch < 1 || block < 1) {
        return;
    }
    const int area = block * block;
    for (int c = 0; c < nch; ++c) {
        if (!nums[c] || !dens[c]) {
            continue;
        }
        unpack_patch(nums[c], dens[c], strides[c], x, y, col + c * area, block, width, height, w);
    }
}

}  // namespace nss
