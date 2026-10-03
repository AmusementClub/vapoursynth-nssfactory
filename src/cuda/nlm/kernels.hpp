// SPDX-License-Identifier: GPL-2.0-only
// Device NLM (the CPU nlm.cpp model): for every offset (frame i, oy, ox) in
// the lower half of the search volume, a clamped distance map between the
// centre frame and the neighbour is box-summed over (2s+1)^2, turned into a
// Welsch weight exp(-sum * h2_inv_norm), and accumulated symmetrically: pixel
// p receives the neighbour at p + o with w1(p) and the mirrored neighbour at
// p - o with w2(p - o). Every pixel is owned by one thread, so results are
// run-to-run identical (D17).
#pragma once

#include <cuda_runtime.h>

namespace nss_cuda {

enum class NlmDistance : int { Luma, Chroma, Yuv, Rgb };  // nss::ChannelMode Y / UV / YUV / RGB

struct NlmPlanes {
    const float* p[3];  // used channels, tight pitch (width floats)
};

// hsum(x, y) = sum_{j=-s..s} D(center(clamp(x+j), y), neighbor(clamp(clamp(x+j)+ox), clamp(y+oy))).
void nlm_distance_hsum(const NlmPlanes& center, const NlmPlanes& neighbor, NlmDistance distance, int ox, int oy,
                       int s, int width, int height, float* hsum, cudaStream_t stream);

struct NlmAccumArgs {
    const float* hsum_bwd;  // pair (centre, backward frame at +o)
    const float* hsum_fwd;  // pair (forward frame, centre at +o); == hsum_bwd for the centre frame
    NlmPlanes src_bwd;
    NlmPlanes src_fwd;
    int channels;
    int ox, oy, s;
    float h2_inv_norm;
    int width, height;
    float* weight;
    float* max_weight;
    float* wdst[3];
};
void nlm_accumulate(const NlmAccumArgs& args, cudaStream_t stream);

// weight = 0, wdst = 0, max_weight = FLT_EPSILON.
void nlm_reset(float* weight, float* max_weight, float* const* wdst, int channels, int width, int height,
               cudaStream_t stream);

// out = (wref * maxw * src + wdst) / (wref * maxw + weight).
void nlm_finish(const float* src, const float* weight, const float* max_weight, const float* wdst, float wref,
                int width, int height, float* out, cudaStream_t stream);

}  // namespace nss_cuda
