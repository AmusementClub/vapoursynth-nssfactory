// SPDX-License-Identifier: GPL-2.0-only
// MCWNNM group filter on the device (docs/vulkan-plan.md V8), the GLSL port
// of nss_cuda's MCWNNM kernels (src/cuda/mcwnnm/kernels.cu, the CPU
// mcwnnm_filter_group model): the three channels of every matched patch are
// stacked into one m x n matrix (m = 3 block^2), rows optionally centred, and
// the channel-weighted WNNM objective is solved by ADMM (X / Z / A iterates,
// rho growing by mu). Each Z step is the WNNM shrinkage of X + A / rho through
// the FP32 Gram matrix; everything after the eigen step is local to a matrix
// row and linear in it, so the kernels iterate on n x n matrices per channel
// class (X = Y P_c, A = Y Q_c) and never store an iterate. Channels of equal
// positive sigma share P, Q and Z (a class, vulkan/mcwnnm/kernels.cpp). Every
// step has a fixed order: run-to-run identical.
#ifndef NSS_VK_MCWNNM_COMMON_GLSL
#define NSS_VK_MCWNNM_COMMON_GLSL

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
// The kernels' state (written and read by one workgroup, and from one kernel
// to the next: coherent, ordered by memoryBarrierBuffer and barrier).
layout(buffer_reference, std430, buffer_reference_align = 4) coherent buffer Scratch { float v[]; };

// The model, constant for a filter (vulkan/mcwnnm/kernels.cpp's ParamsBlock):
// the channel weights (smin / sigma_c)^2 with the smallest positive sigma
// smin, the ADMM's iterations, rho and mu, and the channel classes.
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer Params {
    vec4 weight_smin;  // weight[0..2], smin (user sigma / 255)
    float rho;
    float mu;
    int admm_iter;
    int flags;         // 1: residual (centre the rows), 2: adaptive weight
    int classes;       // 1 to 3
    int class_of[3];   // channel c belongs to class class_of[c]
    int channel_of[3]; // class k has the weight of channel channel_of[k]
    int pad;
};

// As vulkan/mcwnnm/kernels.cpp's Push (128 bytes, the push-constant size
// every Vulkan device has).
layout(push_constant) uniform Push {
    uint64_t src;      // table: window slot t -> the unit's channel 0 (channels channel_step apart)
    uint64_t matches;  // batch * group entries (group-strided)
    uint64_t counts;   // batch
    uint64_t scratch;  // the kernels' state (mcwnnm_scratch_floats a group)
    uint64_t params;   // Params
    FixedTarget fused; // the sums: channel c at fused.num / den + c * fused.channel_step cells
    int channel_step;  // floats from one channel of a source frame to the next
    int pitch;         // floats per row of the planes (and cells per row of the sums)
    int batch;
    int block;
    int group;         // the matched patches a group keeps (<= the kernel's N)
    int pad;
} pc;

Params params() { return Params(pc.params); }
bool mcwnnm_center() { return (params().flags & 1) != 0; }
bool mcwnnm_adaptive() { return (params().flags & 2) != 0; }
float channel_weight(int c) { return params().weight_smin[c]; }
float mcwnnm_smin() { return params().weight_smin.w; }

// Address of the first pixel of match m (channel 0).
uint64_t match_base(Match m) { return Table(pc.src).p[m.t] + uint64_t(m.y * pc.pitch + m.x) * 4ul; }

// The shrinkage constant of n patches (before the ADMM's 2 / rho).
float mcwnnm_constant(int n) {
    const float smin = mcwnnm_smin();
    return 8.0f * sqrt(2.0f * float(n)) * smin * smin;
}

// The weight of a group with `kept` singular values left.
float mcwnnm_weight(int kept) { return mcwnnm_adaptive() && kept > 0 ? 1.0f / float(kept) : 1.0f; }

// The gain of singular value sv at place `place` of the descending order:
// kept as it is below `start`, shrunk below `rank` while s^2 - threshold > 0
// (the spectrum walk of gram_shrink_spectrum: s^2 - threshold falls with s),
// else 0.
float mcwnnm_gain(float sv, int place, int start, int rank, float threshold, out bool kept) {
    kept = place < start;
    if (kept) return 1.0f;
    if (place < rank) {
        const float tmp = sv * sv - threshold;
        if (tmp > 0.0f) {
            kept = true;
            return ((sv + sqrt(tmp)) * 0.5f) / sv;
        }
    }
    return 0.0f;
}

#endif  // NSS_VK_MCWNNM_COMMON_GLSL
