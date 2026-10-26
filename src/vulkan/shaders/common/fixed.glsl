// SPDX-License-Identifier: GPL-2.0-only
// Deterministic aggregation by fixed point (D17, D26): the GLSL side of
// vulkan/common/aggregate.hpp's FixedTarget, as nss_cuda's
// fixed_accumulate.cuh. Cells are int64 sums of value * 2^32 (gpuhost's
// kFixedScale) wrapped in uint64; integer addition is associative and
// commutative, so the 64-bit atomics give the same sums in any order. num
// takes weight * value at every pixel of a patch, den the weight once, at the
// patch's top-left pixel (aggregate.comp's finish sums the block x block
// cells that cover a pixel). A NaN sample adds nothing; values are clamped to
// the int64 range (both far outside video ranges, and deterministic).
//
// The including shader enables GL_EXT_buffer_reference,
// GL_EXT_shader_explicit_arithmetic_types_int64 and GL_EXT_shader_atomic_int64.
#ifndef NSS_VK_FIXED_GLSL
#define NSS_VK_FIXED_GLSL

layout(buffer_reference, std430, buffer_reference_align = 8) buffer FixedCells { uint64_t c[]; };

// As aggregate.hpp's FixedTarget (std430: the int64 members first).
struct FixedTarget {
    uint64_t num;
    uint64_t den;
    int64_t slice_step;    // cells between slices
    int64_t channel_step;  // cells from one channel's slices to the next (joint units)
    int pitch;             // cells per row
    int slice_base;        // a patch of window slot t goes to logical slice t + slice_base
    int slices;            // logical slices [0, slices)
    int slice_first;       // ring cell of logical slice 0
    int slice_ring;        // ring cells
    int pad0, pad1, pad2;
};

const float kFixedScale = 4294967296.0f;  // 2^32 (gpuhost/fixed.hpp)

// Ring cell of the slice of a patch of window slot t, or -1 when the target
// keeps none for it.
int fixed_slice(FixedTarget f, int t) {
    const int slice = t + f.slice_base;
    if (uint(slice) >= uint(f.slices)) return -1;
    const int cell = slice + f.slice_first;
    return cell < f.slice_ring ? cell : cell - f.slice_ring;
}

// First cell of a patch at (x, y) of window slot t, or -1.
int64_t fixed_patch_cell(FixedTarget f, int t, int y, int x) {
    const int slice = fixed_slice(f, t);
    if (slice < 0) return -1l;
    return int64_t(slice) * f.slice_step + int64_t(y) * f.pitch + x;
}

// value in units of 2^-32, rounded to nearest even (__float2ll_rn).
uint64_t fixed_units(float value) {
    const float scaled = value * kFixedScale;  // exact: a power of two
    if (isnan(scaled)) return 0ul;
    const float limit = uintBitsToFloat(0x5effffffu);  // the largest float below 2^63
    return uint64_t(int64_t(roundEven(clamp(scaled, -limit, limit))));
}

void fixed_add(uint64_t cells, int64_t at, float value) {
    atomicAdd(FixedCells(cells + uint64_t(at) * 8ul).c[0], fixed_units(value));
}

#endif  // NSS_VK_FIXED_GLSL
