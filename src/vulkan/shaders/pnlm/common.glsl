// SPDX-License-Identifier: GPL-2.0-only
// PNLM model version 1 (include/nss/params/pnlm.hpp) for the Vulkan kernels:
// the CPU kernel's arithmetic (src/cpu/pnlm/pnlm.cpp) and nss_cuda's
// (src/cuda/pnlm/kernels.cu), operation for operation. Every float operation
// is precise (no contraction, no reassociation), and every quotient is
// correctly rounded (div_rn: Vulkan's division is only 2.5 ULP, D26), so that
// the output can have the CPU's bits.
//
// Defines: Q (2, 3 or 4: patches of P = 2Q at step Q).

#define P (2 * Q)
#define AREA (P * P)

// The plane's parameters, written by the host before the plane's kernels
// (vulkan/pnlm/filter.cpp PnlmParams).
layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer Params {
    uint64_t frames[64];  // the window's frames, this plane: base addresses (pitch floats a row)
    int count;            // frames in the window
    int center;           // the reference frame among them
    int width, height, pitch;
    int a, group;
    int nx, ny;           // the reference grid
    int passes;           // lattice passes of the square matcher (match.comp)
    int pass[16];         // per pass: ox, oy, blocks_x, first_block
    float inv_area, offset, inv_h2, inv_s2;  // nss::PnlmScale
};
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer Plane { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer Floats { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 8) buffer Keys { uint64_t k[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer Ints { int i[]; };

// a / b correctly rounded for the normal operands PNLM divides (Markstein):
// the driver's 1 / b (Vulkan allows 2.5 ULP) refined to the correctly
// rounded reciprocal y by one exact-residual step, then q = a y corrected by
// the exact residual a - q b, twice.
float div_rn(float a, float b) {
    precise float y = 1.0 / b;
    precise float ey = fma(-y, b, 1.0);
    y = fma(ey, y, y);
    precise float q = a * y;
    precise float e = fma(-q, b, a);
    q = fma(e, y, q);
    e = fma(-q, b, a);
    q = fma(e, y, q);
    return q;
}

// Search keys' position fields (nss::pnlm_pos).
uint pnlm_pos(int t, int y, int x) { return uint(t) << 26 | uint(y) << 13 | uint(x); }
int pnlm_pos_t(uint pos) { return int(pos >> 26); }
int pnlm_pos_y(uint pos) { return int((pos >> 13) & 8191u); }
int pnlm_pos_x(uint pos) { return int(pos & 8191u); }

// nss::pnlm_exp.
float pnlm_exp(float x) {
    if (!(x > -87.0)) return 0.0;
    precise float t = x * 1.44269504088896341;
    const float k = roundEven(t);
    precise float f = t - k;
    precise float p = 1.535336188319500e-4;
    p = fma(p, f, 1.339887440266574e-3);
    p = fma(p, f, 9.618437357674640e-3);
    p = fma(p, f, 5.550332471162809e-2);
    p = fma(p, f, 2.402264791363012e-1);
    p = fma(p, f, 6.931472028550421e-1);
    p = fma(p, f, 1.0);
    return ldexp(p, int(k));
}

// nss::pnlm_weight.
float pnlm_weight_of(float dist, float inv_area, float offset, float inv_h2) {
    precise float u = dist * inv_area;
    u = u - offset;
    precise float e = (u > 0.0 ? u : 0.0) * inv_h2;
    return pnlm_exp(-e);
}
float pnlm_weight(float dist, Params pr) { return pnlm_weight_of(dist, pr.inv_area, pr.offset, pr.inv_h2); }

// nss::pnlm_columns: (c0 + c1) + (c2 + c3), (c0 + c1) + c2, c0 + c1.
#if Q == 4
#define PNLM_COLUMNS(c) ((c[0] + c[1]) + (c[2] + c[3]))
#elif Q == 3
#define PNLM_COLUMNS(c) ((c[0] + c[1]) + c[2])
#else
#define PNLM_COLUMNS(c) (c[0] + c[1])
#endif

// nss::pnlm_tent.
float pnlm_tent(int i) {
    const float e = float(2 * i + 1 - P);
    return 1.0 - div_rn(e < 0.0 ? -e : e, float(P));
}

// nss::pnlm_flat: (alpha mean, 1 - alpha) from the sums of the count values
// and of their squares; nss::pnlm_mix.
vec2 pnlm_flat(float s1, float s2, float count, float inv_s2) {
    precise float mean = div_rn(s1, count);
    precise float ex2 = div_rn(s2, count);
    precise float m2 = mean * mean;
    precise float var = ex2 - m2;
    precise float r = (var > 0.0 ? var : 0.0) * inv_s2;
    precise float hi = 1.6 - r;
    // kPnlmFlatHi - kPnlmFlatLo as C++ evaluates it, in float (0.70000005):
    // a shader compiler may fold the literals in double precision (0.7f).
    const float span = uintBitsToFloat(0x3f333334u);
    float alpha = div_rn(hi, span);
    alpha = alpha < 0.0 ? 0.0 : (alpha > 1.0 ? 1.0 : alpha);
    precise float alpha_mean = alpha * mean;
    precise float one_minus = 1.0 - alpha;
    return vec2(alpha_mean, one_minus);
}
float pnlm_mix(float raw, vec2 rule) {
    precise float b = rule.y * raw;
    precise float result = rule.x + b;
    return result;
}

#ifdef PNLM_WEIGH
// The estimate of one reference patch (nss_cuda's weigh): k matches, their
// positions and weights in shared sxy[slot] / sw[slot] (k <= group; dead
// slots have k = 0), by the WARP_L lanes of a segment, lane ll taking the
// patch pixels ll, ll + WARP_L, ... Every lane of the workgroup calls it in
// uniform control flow (the warp layer's rule): the loop runs `group`
// iterations everywhere and entries past k add zeros, which leaves every sum
// bit for bit as it is (no sum here can be -0, so x + 0 == x). Stores the
// estimate at est[ref * AREA] when k > 0.
#define PPL ((AREA + WARP_L - 1) / WARP_L)
void weigh(int k, int slot, int ref, Params pr, Floats est, int ll) {
    precise float acc[PPL];
    precise float s1 = 0.0, s2 = 0.0, wsum = 0.0;
    int offset[PPL];
    bool on[PPL];
    [[unroll]] for (int j = 0; j < PPL; ++j) {
        const int p = ll + j * WARP_L;
        on[j] = p < AREA;
        offset[j] = on[j] ? (p / P) * pr.pitch + p % P : 0;
        acc[j] = 0.0;
    }
    const int group = pr.group;
    for (int i = 0; i < group; ++i) {
        const bool valid = i < k;
        const uint pos = valid ? sxy[slot][i] : 0u;
        const float wt = valid ? sw[slot][i] : 0.0;
        wsum = wsum + wt;
        Plane plane = Plane(pr.frames[pnlm_pos_t(pos)]);
        const int at = pnlm_pos_y(pos) * pr.pitch + pnlm_pos_x(pos);
        [[unroll]] for (int j = 0; j < PPL; ++j) {
            const float v = plane.v[at + offset[j]];
            const float vv = valid && on[j] ? v : 0.0;
            acc[j] = fma(wt, v, acc[j]);
            s1 = s1 + vv;
            s2 = fma(vv, vv, s2);
        }
    }
    s1 = w_sum_f(s1);
    s2 = w_sum_f(s2);
    if (k == 0) return;
    const vec2 rule = pnlm_flat(s1, s2, float(k * AREA), pr.inv_s2);
    [[unroll]] for (int j = 0; j < PPL; ++j) {
        if (!on[j]) continue;
        est.v[ref * AREA + ll + j * WARP_L] = pnlm_mix(div_rn(acc[j], wsum), rule);
    }
}
#endif
