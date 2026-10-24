// SPDX-License-Identifier: GPL-2.0-only
// D23 warp layer: the lane operations of nss_cuda's warp kernels on segments
// of WARP_L lanes (WARP_L = 1, 2, 4, 8, 16 or 32 lanes serve one reference).
//
// WARP_EMULATED 0 (native): subgroup operations. A segment is WARP_L
//   consecutive invocations of one subgroup, so this runs on any subgroup
//   size S that WARP_L divides (S >= WARP_L as both are powers of two): 32 on
//   NVIDIA, 8 on lavapipe for WARP_L <= 8, 64 on GCN. Pipelines require a
//   subgroup size with full subgroups (Pipeline, PipelineDesc::subgroup_size).
// WARP_EMULATED 1: 32-lane warps of consecutive local invocations exchange
//   through shared memory between workgroup barriers; any subgroup size.
//   Every operation must be reached in workgroup-uniform control flow.
// w_any(p) is true whenever p holds on some lane of the segment, and may be
// true otherwise (native: over the whole subgroup, as __any_sync over a warp;
// emulated: always), so callers may use it only to skip work whose result does
// not matter (the results then equal on both layers).
// Both give the same bits: the operations move values, the arithmetic
// around them is the caller's and runs in the same order.
//
// Include after the #version and the definitions of WARP_L, WARP_EMULATED
// and (emulated) WARP_WARPS, the 32-lane warps of a workgroup
// (local_size_x = 32 * WARP_WARPS).
//
// Loops around these operations must unroll in the SPIR-V (constant trip
// count, +-1 or shift steps; spirv-opt --loop-unroll) or be written out: the
// NVIDIA Vulkan compiler keeps such loops rolled (V0 spike, D24 rule).
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_vote : require

#ifndef WARP_L
#error WARP_L must be defined
#endif
#ifndef WARP_EMULATED
#define WARP_EMULATED 0
#endif

#if WARP_EMULATED

#ifndef WARP_WARPS
#error WARP_WARPS must be defined for the emulated warp layer
#endif
shared uvec2 w_xch[WARP_WARPS * 32];

int w_index() { return int(gl_LocalInvocationIndex); }
// Global index of this invocation's segment, and its lane in the segment.
int w_segment() { return int(gl_WorkGroupID.x) * (WARP_WARPS * 32 / WARP_L) + w_index() / WARP_L; }
int w_sub() { return w_index() % WARP_L; }
int w_segment_base() { return w_index() & ~(WARP_L - 1); }

uvec2 w_exchange(uvec2 v, int src_index) {
    barrier();
    w_xch[w_index()] = v;
    barrier();
    return w_xch[src_index];
}

uint w_xor(uint x, int mask) { return w_exchange(uvec2(x, 0u), w_index() ^ mask).x; }
uint w_shfl(uint x, int src) { return w_exchange(uvec2(x, 0u), w_segment_base() + src).x; }
uint64_t w_shfl64(uint64_t x, int src) { return packUint2x32(w_exchange(unpackUint2x32(x), w_segment_base() + src)); }
// __shfl_up_sync(x, 1, L): lane 0 of the segment keeps its own value.
uint64_t w_up64(uint64_t x) {
    return packUint2x32(w_exchange(unpackUint2x32(x), w_sub() == 0 ? w_index() : w_index() - 1));
}
bool w_any(bool p) { return true; }
// Lanes of the segment where p holds, as a mask (bit i: lane i).
uint w_ballot(bool p) {
    barrier();
    w_xch[w_index()] = uvec2(p ? 1u : 0u, 0u);
    barrier();
    uint mask = 0u;
    [[unroll]] for (int i = 0; i < WARP_L; ++i) mask |= w_xch[w_segment_base() + i].x << uint(i);
    return mask;
}

#else

int w_lane() { return int(gl_SubgroupInvocationID); }
int w_segment() {
    return (int(gl_WorkGroupID.x) * int(gl_NumSubgroups) + int(gl_SubgroupID)) * (int(gl_SubgroupSize) / WARP_L) +
           w_lane() / WARP_L;
}
int w_sub() { return w_lane() % WARP_L; }
int w_segment_base() { return w_lane() & ~(WARP_L - 1); }

// Indexed shuffles only (no shuffle_relative, which D22 does not require).
// 64-bit values move as two 32-bit halves: subgroup operations on 64-bit
// types need shaderSubgroupExtendedTypes, which D22 does not require either.
uint w_xor(uint x, int mask) { return subgroupShuffle(x, uint(w_lane() ^ mask)); }
uint w_shfl(uint x, int src) { return subgroupShuffle(x, uint(w_segment_base() + src)); }
uint64_t w_shfl64(uint64_t x, int src) {
    const uvec2 h = unpackUint2x32(x);
    const uint at = uint(w_segment_base() + src);
    return packUint2x32(uvec2(subgroupShuffle(h.x, at), subgroupShuffle(h.y, at)));
}
uint64_t w_up64(uint64_t x) {
    const uvec2 h = unpackUint2x32(x);
    const uint at = uint(max(w_lane() - 1, 0));
    const uint64_t up = packUint2x32(uvec2(subgroupShuffle(h.x, at), subgroupShuffle(h.y, at)));
    return w_sub() == 0 ? x : up;
}
bool w_any(bool p) { return subgroupAny(p); }
uint w_ballot(bool p) {
    const uvec4 b = subgroupBallot(p);
    const int base = w_segment_base();
    const uint word = b[base >> 5] >> uint(base & 31);
    return WARP_L == 32 ? word : word & ((1u << uint(WARP_L)) - 1u);
}

#endif

// Derived operations (both layers).
float w_xor_f(float x, int mask) { return uintBitsToFloat(w_xor(floatBitsToUint(x), mask)); }
float w_shfl_f(float x, int src) { return uintBitsToFloat(w_shfl(floatBitsToUint(x), src)); }
int w_count(bool p) { return bitCount(w_ballot(p)); }

// Sum over the segment as a butterfly (xor masks L/2, ..., 1): every lane
// gets the same sum, the same bits on both layers and from run to run.
// Written out: a shift-stepped loop does not unroll in the SPIR-V.
float w_sum_f(float x) {
#if WARP_L >= 32
    x += w_xor_f(x, 16);
#endif
#if WARP_L >= 16
    x += w_xor_f(x, 8);
#endif
#if WARP_L >= 8
    x += w_xor_f(x, 4);
#endif
#if WARP_L >= 4
    x += w_xor_f(x, 2);
#endif
#if WARP_L >= 2
    x += w_xor_f(x, 1);
#endif
    return x;
}
