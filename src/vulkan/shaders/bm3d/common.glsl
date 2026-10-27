// SPDX-License-Identifier: GPL-2.0-only
// BM3D collaborative filtering on the device (docs/vulkan-plan.md V5), the
// GLSL port of nss_cuda's bm3d kernels (src/cuda/bm3d/kernels.cu). The math
// follows the CPU generic path (Bm3dFilterGroup): orthonormal 2-D DCT per
// patch, orthonormal Haar along the zero-padded group axis of length G
// (noise profile 1, semantic version 3), then the hard
// threshold (|c| < 2.7 sigma -> 0, DC kept, weight 1 / kept) or the Wiener
// shrink against the reference cube (w = r^2 / (r^2 + sigma^2), DC 1, weight
// 1 / sum w^2), the inverse transforms, and the weighted patches added to
// fixed-point sums (common/fixed.glsl; nss_vk aggregates every shape so,
// owner 2026-10-26). The 8-point lines are 4x orthonormal: the forward scales
// the cube by kScale, the round trip by kScale^2, exact powers of two applied
// to the threshold and the result.
//
// Defines: B (block), G (group), WIENER (0 hard threshold, 1 Wiener).
#ifndef NSS_VK_BM3D_COMMON_GLSL
#define NSS_VK_BM3D_COMMON_GLSL

#include "common/dct.glsl"
#include "common/fixed.glsl"

struct Match {
    int x;
    int y;
    int t;
    float dist;
};
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer Plane { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer Table { uint64_t p[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer Matches { Match m[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer Counts { int c[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer Scratch { float v[]; };

// As vulkan/bm3d/kernels.hpp's Bm3dPush (128 bytes, the push-constant size
// every Vulkan device has).
layout(push_constant) uniform Push {
    uint64_t src;      // table: window slot t -> noisy plane
    uint64_t ref;      // table of the Wiener reference planes (WIENER)
    uint64_t matches;  // batch * G entries (group-strided)
    uint64_t counts;   // batch
    uint64_t scratch;  // shape kernel with its cubes in device memory: batch * cube_floats
    int64_t plane_offset;  // floats from a table's plane to the plane filtered (a channel of a unit)
    FixedTarget fused;     // the sums the groups are added to
    int pitch;             // floats per row of the planes (and cells per row of the sums: fused.pitch)
    int batch;
    float sigma;           // effective (profile-scaled)
} pc;

const float kHardLambda = 2.7f;  // nss::kBmHardLambda
#define NSS_LINE_SCALE(n) ((n) == 8 ? 4.0f : 1.0f)  // dct_line_scale, as a constant expression
const float kScale = NSS_LINE_SCALE(B) * NSS_LINE_SCALE(B);  // the group-axis Haar is orthonormal
const float kUnscale = 1.0f / (kScale * kScale);

#define NSS_CAT(a, b) a##b
#define DCT_LINE(n) NSS_CAT(dct_line, n)
#define HAAR_LINE(n) NSS_CAT(haar_line, n)

// The group-axis transform: the orthonormal Haar transform of n samples in
// the CPU's in-place layout (HaarLines in cpu/bm/dct8.cpp): level h pairs the
// samples h apart (sum to the lower, difference to the upper), the DC stays
// at sample 0, and each position's factor (2^(-1/2) per level it went
// through) is applied once: last forward, first inverse.
float haar_scale(int n, int j) {
    int e = 0;
    if (j == 0) {
        while ((1 << e) < n) ++e;
    } else {
        while (((j >> e) & 1) == 0) ++e;
        ++e;
    }
    float f = 1.0f;
    for (int i = 0; i < e / 2; ++i) f *= 0.5f;
    return (e % 2) != 0 ? f * 0.707106781186547524f : f;
}

void haar_line1(inout float v[1], bool inverse) {}

#define NSS_HAAR_LINE(N)                                                                  \
    void haar_line##N(precise inout float v[N], bool inverse) {                           \
        if (inverse) {                                                                    \
            [[unroll]] for (int j = 0; j < N; ++j) v[j] *= haar_scale(N, j);              \
            [[unroll]] for (int h = N / 2; h >= 1; h >>= 1) {                             \
                [[unroll]] for (int i = 0; i < N; i += 2 * h) {                           \
                    const float a = v[i], b = v[i + h];                                   \
                    v[i] = a + b;                                                         \
                    v[i + h] = a - b;                                                     \
                }                                                                         \
            }                                                                             \
        } else {                                                                          \
            [[unroll]] for (int h = 1; h < N; h <<= 1) {                                  \
                [[unroll]] for (int i = 0; i < N; i += 2 * h) {                           \
                    const float a = v[i], b = v[i + h];                                   \
                    v[i] = a + b;                                                         \
                    v[i + h] = a - b;                                                     \
                }                                                                         \
            }                                                                             \
            [[unroll]] for (int j = 0; j < N; ++j) v[j] *= haar_scale(N, j);              \
        }                                                                                 \
    }
NSS_HAAR_LINE(2)
NSS_HAAR_LINE(4)
NSS_HAAR_LINE(8)
NSS_HAAR_LINE(16)
NSS_HAAR_LINE(32)
NSS_HAAR_LINE(64)

// Address of pixel (x, y) of the plane of window slot t in table `table`.
uint64_t plane_base(uint64_t table, int t) { return Table(table).p[t] + uint64_t(pc.plane_offset) * 4ul; }

#endif  // NSS_VK_BM3D_COMMON_GLSL
