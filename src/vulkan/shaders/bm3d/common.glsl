// SPDX-License-Identifier: GPL-2.0-only
// BM3D collaborative filtering on the device (docs/vulkan-plan.md V5), the
// GLSL port of nss_cuda's bm3d kernels (src/cuda/bm3d/kernels.cu). The math
// follows the CPU generic path (Bm3dFilterGroup): orthonormal 2-D DCT per
// patch, DCT along the zero-padded group axis of length G, then the hard
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
const float kScale = NSS_LINE_SCALE(B) * NSS_LINE_SCALE(B) * NSS_LINE_SCALE(G);
const float kUnscale = 1.0f / (kScale * kScale);

#define NSS_CAT(a, b) a##b
#define DCT_LINE(n) NSS_CAT(dct_line, n)

// Address of pixel (x, y) of the plane of window slot t in table `table`.
uint64_t plane_base(uint64_t table, int t) { return Table(table).p[t] + uint64_t(pc.plane_offset) * 4ul; }

#endif  // NSS_VK_BM3D_COMMON_GLSL
