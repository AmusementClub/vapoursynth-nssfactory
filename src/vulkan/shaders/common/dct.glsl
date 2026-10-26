// SPDX-License-Identifier: GPL-2.0-only
// DCT-II / DCT-III lines for the patch-group shaders (docs/vulkan-plan.md V3),
// the GLSL port of nss_cuda's dct_line_fixed<N> (src/cuda/bm3d/kernels.cu):
// - N = 8: the FFTW e10_8 / e01_8 butterflies of the CPU's bm3d_filter8, each
//   4x the orthonormal transform (dct_line_scale);
// - N = 2, 4: the orthonormal matrix product the CPU uses (same terms in the
//   same order);
// - N = 12, 16, 32, 64: the FFTW genfft codelets the CPU compiles through
//   Highway (orthonormal), converted to GLSL at configure time
//   (shaders_gen/common/dct_codelet_*.glsl, src/vulkan/CMakeLists.txt).
// Every intermediate is precise and every fused step an explicit fma() (D26):
// on a driver that fuses fma (NVIDIA) the result is the arithmetic the CPU
// source writes (tests/vk/test_vk_aggregate.cpp checks it bit for bit against
// a host model of it). That is not a promise of the CPU's or CUDA's bits: the
// CPU's 8-point file is built with -ffast-math and nss_cuda with fmad, either
// of which may fuse the one multiply-subtract of dct8_inverse (tk). Mesa splits
// fma even under precise (the last bit may differ).
//
// dct_line<N>(v, inverse) transforms the N values of v in place; the caller
// gathers a line (any stride) into v and scatters it back.
#ifndef NSS_VK_DCT_GLSL
#define NSS_VK_DCT_GLSL

const float kDctP414 = 0.414213562373095048801688724209698078569671875;
const float kDctP1847 = 1.847759065022573512256366378793576573644833252;
const float kDctP198 = 0.198912367379658006911597622644676228597850501;
const float kDctP1961 = 1.961570560806460898252364472268478073947867462;
const float kDctP1414 = 1.414213562373095048801688724209698078569671875;
const float kDctP668 = 0.668178637919298919997757686523080761552472251;
const float kDctP1662 = 1.662939224605090474157576755235811513477121624;
const float kDctP707 = 0.707106781186547524400844362104849039284835938;
const float kDct4A = 0.653281482438188263928322;  // cos(pi / 8) / sqrt(2)
const float kDct4B = 0.270598050073098492199862;  // sin(pi / 8) / sqrt(2)

// Scale of dct_line<N> against the orthonormal transform.
float dct_line_scale(int n) { return n == 8 ? 4.0f : 1.0f; }

void dct8_forward(precise inout float b[8]) {
    precise float t1 = b[0], t2 = b[7];
    precise float t3 = t1 - t2, tj = t1 + t2;
    precise float tc = b[4], td = b[3];
    precise float te = tc - td, tk = tc + td;
    precise float t4 = b[2], t5 = b[5];
    precise float t6 = t4 - t5;
    precise float t7 = b[1], t8 = b[6];
    precise float t9 = t7 - t8;
    precise float ta = t6 + t9, tn = t7 + t8, tf = t6 - t9, tm = t4 + t5;
    precise float tb = fma(-kDctP707, ta, t3);
    precise float tg = fma(-kDctP707, tf, te);
    b[3] = kDctP1662 * fma(kDctP668, tg, tb);
    b[5] = -(kDctP1662 * fma(-kDctP668, tb, tg));
    precise float tp = tj + tk, tq = tm + tn;
    b[4] = kDctP1414 * (tp - tq);
    b[0] = kDctP1414 * (tp + tq);
    precise float th = fma(kDctP707, ta, t3);
    precise float ti = fma(kDctP707, tf, te);
    b[1] = kDctP1961 * fma(-kDctP198, ti, th);
    b[7] = kDctP1961 * fma(kDctP198, th, ti);
    precise float tl = tj - tk, to = tm - tn;
    b[2] = kDctP1847 * fma(-kDctP414, to, tl);
    b[6] = kDctP1847 * fma(kDctP414, tl, to);
}

void dct8_inverse(precise inout float b[8]) {
    precise float t1 = kDctP1414 * b[0];
    precise float t2 = b[4];
    precise float t3 = fma(kDctP1414, t2, t1);
    precise float tj = fma(-kDctP1414, t2, t1);
    precise float t4 = b[2], t5 = b[6];
    precise float t6 = fma(kDctP414, t5, t4);
    precise float tk = kDctP414 * t4 - t5;
    precise float t8 = b[1], td = b[7], t9 = b[5], ta = b[3];
    precise float tb = t9 + ta, te = ta - t9;
    precise float tc = fma(kDctP707, tb, t8);
    precise float tn = fma(-kDctP707, te, td);
    precise float tf = fma(kDctP707, te, td);
    precise float tm = fma(-kDctP707, tb, t8);
    precise float t7 = fma(kDctP1847, t6, t3);
    precise float tg = fma(kDctP198, tf, tc);
    b[7] = fma(-kDctP1961, tg, t7);
    b[0] = fma(kDctP1961, tg, t7);
    precise float tp = fma(-kDctP1847, tk, tj);
    precise float tq = fma(kDctP668, tm, tn);
    b[5] = fma(-kDctP1662, tq, tp);
    b[2] = fma(kDctP1662, tq, tp);
    precise float th = fma(-kDctP1847, t6, t3);
    precise float ti = fma(-kDctP198, tc, tf);
    b[3] = fma(-kDctP1961, ti, th);
    b[4] = fma(kDctP1961, ti, th);
    precise float tl = fma(kDctP1847, tk, tj);
    precise float to = fma(-kDctP668, tn, tm);
    b[6] = fma(-kDctP1662, to, tl);
    b[1] = fma(kDctP1662, to, tl);
}

// Orthonormal 2- and 4-point DCT-II (row k, column j) as the CPU's product.
const float kDct2[2][2] = {{kDctP707, kDctP707}, {kDctP707, -kDctP707}};
const float kDct4[4][4] = {{0.5f, 0.5f, 0.5f, 0.5f},
                           {kDct4A, kDct4B, -kDct4B, -kDct4A},
                           {0.5f, -0.5f, -0.5f, 0.5f},
                           {kDct4B, -kDct4A, kDct4A, -kDct4B}};

void dct_line1(inout float v[1], bool inverse) {}

void dct_line2(precise inout float v[2], bool inverse) {
    float in_[2] = v;
    [[unroll]] for (int k = 0; k < 2; ++k) {
        precise float acc = 0.0f;
        [[unroll]] for (int j = 0; j < 2; ++j) acc = fma(inverse ? kDct2[j][k] : kDct2[k][j], in_[j], acc);
        v[k] = acc;
    }
}

void dct_line4(precise inout float v[4], bool inverse) {
    float in_[4] = v;
    [[unroll]] for (int k = 0; k < 4; ++k) {
        precise float acc = 0.0f;
        [[unroll]] for (int j = 0; j < 4; ++j) acc = fma(inverse ? kDct4[j][k] : kDct4[k][j], in_[j], acc);
        v[k] = acc;
    }
}

void dct_line8(precise inout float v[8], bool inverse) {
    if (inverse) {
        dct8_inverse(v);
    } else {
        dct8_forward(v);
    }
}

// The genfft codelets' vector macros, mapped to precise scalar floats.
#define R float
#define V precise float
#define DVK(name, value) const float name = float(value)
#define LDK(name) (name)
#define LD(p, vstride, alignment) (p)
#define ST(p, value, vstride, alignment) p = (value)
#define VADD(a, b) ((a) + (b))
#define VSUB(a, b) ((a) - (b))
#define VMUL(a, b) ((a) * (b))
#define VFMA(a, b, c) fma((a), (b), (c))
#define VFMS(a, b, c) fma((a), (b), -(c))
#define VFNMS(a, b, c) fma(-(a), (b), (c))
#define VNEG(a) (-(a))
#define VLEAVE()
#include "common/dct_codelet_fwd_n12.glsl"
#include "common/dct_codelet_inv_n12.glsl"
#include "common/dct_codelet_fwd_n16.glsl"
#include "common/dct_codelet_inv_n16.glsl"
#include "common/dct_codelet_fwd_n32.glsl"
#include "common/dct_codelet_inv_n32.glsl"
#include "common/dct_codelet_fwd_n64.glsl"
#include "common/dct_codelet_inv_n64.glsl"
#undef R
#undef V
#undef DVK
#undef LDK
#undef LD
#undef ST
#undef VADD
#undef VSUB
#undef VMUL
#undef VFMA
#undef VFMS
#undef VFNMS
#undef VNEG
#undef VLEAVE

#define NSS_VK_DCT_CODELET_LINE(N)                         \
    void dct_line##N(precise inout float v[N], bool inverse) { \
        float o[N];                                        \
        if (inverse) {                                     \
            nss_dct##N##_inv_is8(v, o);                    \
        } else {                                           \
            nss_dct##N##_fwd_is8(v, o);                    \
        }                                                  \
        v = o;                                             \
    }
NSS_VK_DCT_CODELET_LINE(12)
NSS_VK_DCT_CODELET_LINE(16)
NSS_VK_DCT_CODELET_LINE(32)
NSS_VK_DCT_CODELET_LINE(64)
#undef NSS_VK_DCT_CODELET_LINE

#endif  // NSS_VK_DCT_GLSL
