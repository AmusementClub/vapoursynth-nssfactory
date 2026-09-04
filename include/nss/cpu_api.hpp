#pragma once

#include "nss/params.hpp"
#include "nss/plane.hpp"

#include <cstddef>
#include <cstdint>

namespace nss {

enum class ChannelMode { Y, UV, YUV, RGB };

void nlm_distance_luma_f32(float* dst, const float* center, const float* neighbor,
                           int ox, int oy, int w, int h, int stride);
#if defined(__GNUC__) || defined(__clang__)
__attribute__((visibility("hidden")))
#endif
void nlm_distance_luma_horizontal_f32(float* dst, float* row_scratch, const float* center,
                                      const float* neighbor, int ox, int oy, int radius,
                                      int w, int h, int stride);
void nlm_distance_chroma_f32(float* dst, const float* c1, const float* c2,
                             const float* n1, const float* n2,
                             int ox, int oy, int w, int h, int stride);
void nlm_distance_yuv_f32(float* dst, const float* c0, const float* c1, const float* c2,
                          const float* n0, const float* n1, const float* n2,
                          int ox, int oy, int w, int h, int stride);
void nlm_distance_rgb_f32(float* dst, const float* c0, const float* c1, const float* c2,
                          const float* n0, const float* n1, const float* n2,
                          int ox, int oy, int w, int h, int stride);
// Plane-stride aware distance path used when a multi-plane frame does not
// expose one common row stride. The destination is a packed map with
// `dst_stride`; source pointers are row-zero bases for the local stripe.
void nlm_distance_strided_f32(float* dst, const float* const* center, const int* center_strides,
                              const float* const* neighbor, const int* neighbor_strides,
                              ChannelMode mode, int ox, int oy, int w, int h, int dst_stride);
void nlm_horizontal(float* dst, const float* src, int s, int w, int h, int stride);
void nlm_vertical_welsch(float* dst, const float* src, int radius, float h2_inv_norm,
                         int w, int h, int stride, float* buffer);
// Writes only source rows [y0, y1) to a compact destination whose row zero is y0.
void nlm_vertical_welsch_range(float* dst, const float* src, int radius, float h2_inv_norm,
                               int w, int h, int stride, int y0, int y1, float* buffer);
void nlm_accum_ch1(float* weight, float* wdst, float* maxw,
                   const float* src_bwd, const float* src_fwd,
                   const float* temp1, const float* temp2,
                   int ox, int oy, int w, int h, int stride);
void nlm_accum_ch2(float* weight, float* wdst0, float* wdst1, float* maxw,
                   const float* s0_bwd, const float* s1_bwd,
                   const float* s0_fwd, const float* s1_fwd,
                   const float* temp1, const float* temp2,
                   int ox, int oy, int w, int h, int stride);
void nlm_accum_ch3(float* weight, float* wdst0, float* wdst1, float* wdst2, float* maxw,
                   const float* s0_bwd, const float* s1_bwd, const float* s2_bwd,
                   const float* s0_fwd, const float* s1_fwd, const float* s2_fwd,
                   const float* temp1, const float* temp2,
                   int ox, int oy, int w, int h, int stride);
// Range variants keep the distance/box buffers in an extended stripe while
// writing accumulation state only for [y0, y1).  `h` remains the full stripe
// height so replicated border and mirrored temporal coordinates are unchanged.
void nlm_accum_ch1_range(float* weight, float* wdst, float* maxw,
                         const float* src_bwd, const float* src_fwd,
                         const float* temp1, const float* temp2,
                         int ox, int oy, int w, int h, int stride, int y0, int y1);
void nlm_accum_ch2_range(float* weight, float* wdst0, float* wdst1, float* maxw,
                         const float* s0_bwd, const float* s1_bwd,
                         const float* s0_fwd, const float* s1_fwd,
                         const float* temp1, const float* temp2,
                         int ox, int oy, int w, int h, int stride, int y0, int y1);
void nlm_accum_ch3_range(float* weight, float* wdst0, float* wdst1, float* wdst2, float* maxw,
                         const float* s0_bwd, const float* s1_bwd, const float* s2_bwd,
                         const float* s0_fwd, const float* s1_fwd, const float* s2_fwd,
                         const float* temp1, const float* temp2,
                         int ox, int oy, int w, int h, int stride, int y0, int y1);
void nlm_accum_ch1_core_range(float* weight, float* wdst, float* maxw,
                              const float* src_bwd, const float* src_fwd,
                              const float* temp1_core, const float* temp2,
                              int ox, int oy, int w, int h, int stride, int y0, int y1, int temp2_base_y = 0);
void nlm_accum_ch2_core_range(float* weight, float* wdst0, float* wdst1, float* maxw,
                              const float* s0_bwd, const float* s1_bwd,
                              const float* s0_fwd, const float* s1_fwd,
                              const float* temp1_core, const float* temp2,
                              int ox, int oy, int w, int h, int stride, int y0, int y1, int temp2_base_y = 0);
void nlm_accum_ch3_core_range(float* weight, float* wdst0, float* wdst1, float* wdst2, float* maxw,
                              const float* s0_bwd, const float* s1_bwd, const float* s2_bwd,
                              const float* s0_fwd, const float* s1_fwd, const float* s2_fwd,
                              const float* temp1_core, const float* temp2,
                              int ox, int oy, int w, int h, int stride, int y0, int y1, int temp2_base_y = 0);
// Generic scalar fallback for mixed source-plane strides. `temp1_base_y` is
// zero for a full/range map and y0 for a core-only temp1 map.
void nlm_accum_strided(float* weight, float* wdst0, float* wdst1, float* wdst2, float* maxw,
                       const float* const* src_bwd, const int* src_bwd_strides,
                       const float* const* src_fwd, const int* src_fwd_strides,
                       const float* temp1, const float* temp2, int nch,
                       int ox, int oy, int w, int h, int stride, int y0, int y1, int temp1_base_y);
void nlm_finish_ch1(float* dst, const float* src, const float* weight, const float* wdst,
                    const float* maxw, float wref, int w, int h, int stride);
void nlm_finish_ch2(float* d0, float* d1, const float* s0, const float* s1,
                    const float* weight, const float* wdst0, const float* wdst1,
                    const float* maxw, float wref, int w, int h, int stride);
void nlm_finish_ch3(float* d0, float* d1, float* d2,
                    const float* s0, const float* s1, const float* s2,
                    const float* weight, const float* wdst0, const float* wdst1, const float* wdst2,
                    const float* maxw, float wref, int w, int h, int stride);
// Plane-stride aware finish path. Source and destination pointers are
// row-zero bases for the output core, while accumulation maps use `map_stride`.
void nlm_finish_strided(float* const* dst, const int* dst_strides, const float* const* src,
                        const int* src_strides, const float* weight, const float* const* wdst,
                        const float* maxw, float wref, int nch, int w, int h, int map_stride);

void dct_1d(const float* in, float* out, int n);
void idct_1d(const float* in, float* out, int n);
void dct_2d(float* block, int n);
void idct_2d(float* block, int n);
void dct_lines(float* base, int n, int line_stride, int sample_stride, int count, bool inverse);
void dct8_1d(const float in[8], float out[8]);
void idct8_1d(const float in[8], float out[8]);
void dct8_2d(float block[64]);
void idct8_2d(float block[64]);

float ssd_block(const float* a, int sa, const float* b, int sb, int block);
float ssd_nch(const float* const* a, const int* sa, const float* const* b, const int* sb, int nch, int block);

struct Match {
    int x = 0;
    int y = 0;
    int t = 0;
    float dist = 0.f;
    // Candidate traversal order makes equal-distance selection deterministic.
    std::uint32_t ordinal = 0;
};

int spatial_match(const float* ref, int stride, int width, int height,
                  int bx, int by, int block, int bm_range, int group,
                  Match* out);

int predictive_match(const float* const* refs, const int* strides, int ntemp,
                     int width, int height, int bx, int by, int t0,
                     const SearchConfig& cfg, Match* out);

void pack_patch(float* col, int lda, const float* src, int stride, int x, int y,
                int block, int width, int height);
void unpack_patch(float* num, float* den, int stride, int x, int y,
                  const float* col, int block, int width, int height, float w);

inline int bm3d_filter_work_floats(int group, int block) {
    return 2 * group * block * block;
}

// `k` real patches in patches[0..k); `group` is the zero-padded 3D-transform length.
// `work` must hold bm3d_filter_work_floats(group, block) floats.
void bm3d_filter_group(float* patches, int lda, int group, int k, int block, float sigma, bool wiener,
                       const float* ref_patches, float* weight_out, float* work);

// 8x8x8 fused path: in-register FFTW 3D DCT (bm3dcpu layout), shrink, accumulate. k in [1, 8].
void bm3d_filter8(const float* src, int sstride, const Match* matches, int k, float sigma, bool wiener,
                  const float* ref, int rstride, float* num, float* den, int dstride, int width, int height);

// Load from the source image, orthonormal bm3d_filter_group, accumulate. Not the 8x8x8 FFTW path.
// `cube` holds group*block*block floats, plus another group*block*block if wiener.
// `work` holds bm3d_filter_work_floats(group, block) floats.
void bm3d_filter_direct(const float* src, int sstride, const Match* matches, int k, int block, int group, float sigma,
                        bool wiener, const float* ref, int rstride, float* num, float* den, int dstride, int width,
                        int height, float* cube, float* work);

void aggregate_add(float* num, float* den, int stride, int x, int y,
                   const float* patch, int block, int width, int height, float w);
// sstride < 1 means src uses dstride (Wave 1 callers). dst uses dstride; num/den use buf_stride.
void aggregate_finish(float* dst, const float* num, const float* den, const float* src,
                      int width, int height, int dstride, int buf_stride, int sstride = -1);

int svd_economy(int m, int n, const float* A, int lda,
                float* U, int ldu, float* S, float* Vt, int ldvt,
                float* work = nullptr, int work_floats = 0);

inline int wnnm_shrink_work_floats(int m, int n) {
    const int sh = m * n + n + n * n + m;
    const int svd = m * n * 6 + n * n * 8 + n + 256;
    return sh + svd;
}

int wnnm_shrink(float* group, int m, int n, int lda, float sigma,
                int residual, int adaptive, float* adaptive_weight,
                float* work, int work_floats);

void vaggregate_reduce(float* dst, const float* fat, const float* src,
                       int width, int height, int dstride, int fstride, int radius);

}  // namespace nss
