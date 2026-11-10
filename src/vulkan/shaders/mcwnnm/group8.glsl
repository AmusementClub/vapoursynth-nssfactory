// SPDX-License-Identifier: GPL-2.0-only
// MCWNNM, groups of up to 8 patches (nss_cuda's mcwnnm_rows_kernel,
// mcwnnm_group_kernel<8> and mcwnnm_apply_kernel): rows8, 8 invocations a
// group, the channel classes' Gram matrices of the (centred) input; admm8,
// one invocation a group, the ADMM on n x n matrices; apply8, the layout of
// the first again, the output Y_c Z_c into the sums. The state is
// interleaved over chunks of 64 groups (element e of group r at e * width +
// r % 64, width the groups of the chunk), so that the ADMM's invocations and
// the 8-invocation groups both touch consecutive words.
#ifndef NSS_VK_MCWNNM_GROUP8_GLSL
#define NSS_VK_MCWNNM_GROUP8_GLSL

#include "mcwnnm/common.glsl"

const int kN = 8;
const int kChunk = 64;
// Matrices of the state, per class c: (matrix * 3 + c) group^2 floats in;
// kVectors: the last eigen step's vectors (class 0's place only); the
// aggregation weight after the ten (vulkan/mcwnnm/kernels.cpp).
const int kGram = 0, kP = 1, kQ = 2, kVectors = 3, kWeight = 10;

int state_square() { return pc.group * pc.group; }
int state_floats() { return 10 * state_square() + 1; }

// Element 0 of group r's state, and the stride of its elements.
uint64_t state_at(int r) {
    const int chunk = r / kChunk;
    return pc.scratch + (uint64_t(chunk) * uint64_t(kChunk) * uint64_t(state_floats()) + uint64_t(r % kChunk)) * 4ul;
}
int state_width(int r) { return min(kChunk, pc.batch - r / kChunk * kChunk); }

// Element (j, k) of matrix `matrix` of class c, in floats from the state's element 0.
int state_index(int matrix, int c, int j, int k, int width) {
    return ((matrix * 3 + c) * state_square() + j * pc.group + k) * width;
}

// The patches' first pixels (column j of the group; beyond n the first).
void group_columns(int r, int n, out uint64_t col[kN]) {
    [[unroll]] for (int j = 0; j < kN; ++j) col[j] = match_base(Matches(pc.matches).m[r * pc.group + (j < n ? j : 0)]);
}

// Pixel `offset` (from channel 0) of the group's patches, centred (residual)
// on its mean, which is returned (0 without residual); zero beyond n.
float group_pixel(uint64_t col[kN], int offset, int n, out float value[kN]) {
    float sum = 0.0f;
    [[unroll]] for (int j = 0; j < kN; ++j) {
        value[j] = j < n ? Plane(col[j]).v[offset] : 0.0f;
        sum += value[j];
    }
    const float mean = mcwnnm_center() ? sum * (1.0f / float(n)) : 0.0f;
    [[unroll]] for (int j = 0; j < kN; ++j) value[j] = j < n ? value[j] - mean : 0.0f;
    return mean;
}

#endif  // NSS_VK_MCWNNM_GROUP8_GLSL
