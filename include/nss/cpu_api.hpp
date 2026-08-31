#pragma once

#include "nss/params.hpp"
#include "nss/plane.hpp"

#include <cstddef>

namespace nss {

enum class ChannelMode { Y, UV, YUV, RGB };

void nlm_distance_luma_f32(float* dst, const float* center, const float* neighbor,
                           int ox, int oy, int w, int h, int stride);
void nlm_distance_chroma_f32(float* dst, const float* c1, const float* c2,
                             const float* n1, const float* n2,
                             int ox, int oy, int w, int h, int stride);
void nlm_distance_yuv_f32(float* dst, const float* c0, const float* c1, const float* c2,
                          const float* n0, const float* n1, const float* n2,
                          int ox, int oy, int w, int h, int stride);
void nlm_distance_rgb_f32(float* dst, const float* c0, const float* c1, const float* c2,
                          const float* n0, const float* n1, const float* n2,
                          int ox, int oy, int w, int h, int stride);
void nlm_horizontal(float* dst, const float* src, int s, int w, int h, int stride);
void nlm_vertical_welsch(float* dst, const float* src, int radius, float h2_inv_norm,
                         int w, int h, int stride, float* buffer);
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
void nlm_finish_ch1(float* dst, const float* src, const float* weight, const float* wdst,
                    const float* maxw, float wref, int w, int h, int stride);
void nlm_finish_ch2(float* d0, float* d1, const float* s0, const float* s1,
                    const float* weight, const float* wdst0, const float* wdst1,
                    const float* maxw, float wref, int w, int h, int stride);
void nlm_finish_ch3(float* d0, float* d1, float* d2,
                    const float* s0, const float* s1, const float* s2,
                    const float* weight, const float* wdst0, const float* wdst1, const float* wdst2,
                    const float* maxw, float wref, int w, int h, int stride);

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

struct Match {
    int x = 0;
    int y = 0;
    int t = 0;
    float dist = 0.f;
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
