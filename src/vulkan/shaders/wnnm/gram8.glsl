// SPDX-License-Identifier: GPL-2.0-only
// Groups of up to 8 patches (nss_cuda's three Gram kernels, gram_group.cuh):
// gram_rows, 8 invocations a group, invocation l taking the pixels of column
// l of every patch row (a row of a patch is then one request of the group's
// invocations), summing their Gram products, then a butterfly over the 8
// (ascending masks, as nss_cuda's shuffle tree); gram_solve, one invocation a
// group, the eigen step and the WNNM finish in registers; gram_apply, the
// layout of the first again, X = A M pixel by pixel into the sums. The
// scratch is interleaved over chunks of 64 groups (element e of group r at
// e * width + r % 64, width the groups of the chunk), so that the solve
// kernel's invocations and the 8-invocation groups both touch consecutive
// words.
#ifndef NSS_VK_WNNM_GRAM8_GLSL
#define NSS_VK_WNNM_GRAM8_GLSL

#include "wnnm/common.glsl"

const int kN = 8;
const int kChunk = 64;
const int kSplitGram = 36;                   // the Gram matrix's lower triangle
const int kSplitFloats = kSplitGram + 64 + 1;  // then M and the weight (vulkan/wnnm/kernels.cpp)

// Element 0 of group r's scratch, and the stride of its elements.
uint64_t split_at(int r) {
    const int chunk = r / kChunk;
    return pc.scratch + uint64_t(chunk * kChunk * kSplitFloats + r % kChunk) * 4ul;
}
int split_width(int r) { return min(kChunk, pc.batch - r / kChunk * kChunk); }

// The patches' first pixels (column j of the group; beyond n the first).
void group_columns(int r, int n, out uint64_t col[kN]) {
    [[unroll]] for (int j = 0; j < kN; ++j) col[j] = match_base(Matches(pc.matches).m[r * pc.group + (j < n ? j : 0)]);
}

// Pixel `offset` of the group's patches, centred (residual) on its mean, which
// is returned (0 without residual); zero beyond n.
float group_pixel(uint64_t col[kN], int offset, int n, out float value[kN]) {
    float sum = 0.0f;
    [[unroll]] for (int j = 0; j < kN; ++j) {
        value[j] = j < n ? Plane(col[j]).v[offset] : 0.0f;
        sum += value[j];
    }
    const float mu = wnnm_center() ? sum / float(n) : 0.0f;
    [[unroll]] for (int j = 0; j < kN; ++j) value[j] = j < n ? value[j] - mu : 0.0f;
    return mu;
}

#endif  // NSS_VK_WNNM_GRAM8_GLSL
