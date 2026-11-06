// SPDX-License-Identifier: GPL-2.0-only
// WNNM group shrinkage on the device (docs/vulkan-plan.md V7), the GLSL port
// of nss_cuda's Gram group kernels (src/cuda/common/gram_group.cuh) with the
// WNNM model (src/cuda/wnnm/kernels.cu): the m x n group matrix A (m = block^2
// rows, n matched patches as columns), with residual = 1 each row centred on
// its mean over the group; the singular values of A through its Gram matrix
// A^T A = V S^2 V^T, shrunk by (s + sqrt(s^2 - c)) / 2 with
// c = 8 sqrt(2n) sigma^2 walking the ordered spectrum (the first one kept
// without residual), X = A V diag(s'/s) V^T, and the patches added to the
// fixed-point sums with weight 1 / kept (adaptive) or 1 (common/fixed.glsl).
// Every step has a fixed order: run-to-run identical. As nss_cuda's (built
// with contraction) this is close to the CPU's output, not the same bits.
#ifndef NSS_VK_WNNM_COMMON_GLSL
#define NSS_VK_WNNM_COMMON_GLSL

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
// The kernels' scratch (written and read by the invocations of a workgroup:
// coherent, ordered by memoryBarrierBuffer and barrier).
layout(buffer_reference, std430, buffer_reference_align = 4) coherent buffer Scratch { float v[]; };

// As vulkan/wnnm/kernels.cpp's Push (128 bytes, the push-constant size every
// Vulkan device has).
layout(push_constant) uniform Push {
    uint64_t src;      // table: window slot t -> plane
    uint64_t matches;  // batch * group entries (group-strided)
    uint64_t counts;   // batch
    uint64_t scratch;  // the kernel's scratch (wnnm_scratch_floats a group)
    FixedTarget fused; // the sums the groups are added to
    int pitch;         // floats per row of the planes (and cells per row of the sums)
    int batch;
    int block;
    int group;         // the matched patches a group keeps (<= the kernel's N)
    float sigma;       // user sigma / 255
    int flags;         // 1: residual (centre the rows), 2: adaptive weight
} pc;

bool wnnm_center() { return (pc.flags & 1) != 0; }
bool wnnm_adaptive() { return (pc.flags & 2) != 0; }

// Address of the first pixel of match m.
uint64_t match_base(Match m) { return Table(pc.src).p[m.t] + uint64_t(m.y * pc.pitch + m.x) * 4ul; }

// The shrinkage constant of n patches.
float wnnm_constant(int n) { return 8.0f * sqrt(2.0f * float(n)) * pc.sigma * pc.sigma; }

// The weight of a group with `kept` singular values left.
float wnnm_weight(int kept) { return wnnm_adaptive() && kept > 0 ? 1.0f / float(kept) : 1.0f; }

// The gain of singular value sv at place `place` of the descending order:
// kept as it is below `start`, shrunk below `rank` while s^2 - c > 0, else 0.
float wnnm_gain(float sv, int place, int start, int rank, float constant, out bool kept) {
    kept = place < start;
    if (kept) return 1.0f;
    if (place < rank) {
        const float tmp = sv * sv - constant;
        if (tmp > 0.0f) {
            kept = true;
            return ((sv + sqrt(tmp)) * 0.5f) / sv;
        }
    }
    return 0.0f;
}

#endif  // NSS_VK_WNNM_COMMON_GLSL
