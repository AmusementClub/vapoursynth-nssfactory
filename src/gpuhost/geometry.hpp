// SPDX-License-Identifier: GPL-2.0-only
// Block-matching geometry shared by the GPU backends (D27): the CPU raster
// grid of reference positions and the shape of a search. Backend-free; under
// nvcc the member functions are also device functions.
#pragma once

#if defined(__CUDACC__)
#define NSS_GPUHOST_HD __host__ __device__
#else
#define NSS_GPUHOST_HD
#endif

namespace nss::gpuhost {

// Reference positions: the CPU raster grid (append_raster_jobs) with the last
// row/column clamped to (width - block, height - block).
struct RasterGrid {
    int nx;
    int ny;
    int step;
    int max_x;
    int max_y;
    NSS_GPUHOST_HD int count() const { return nx * ny; }
    NSS_GPUHOST_HD int x(int i) const { const int v = (i % nx) * step; return v < max_x ? v : max_x; }
    NSS_GPUHOST_HD int y(int i) const { const int v = (i / nx) * step; return v < max_y ? v : max_y; }
};
RasterGrid make_raster_grid(int width, int height, int block, int step);

struct MatchGeometry {
    int width;
    int height;
    int pitch;  // in floats
    int block;
    int bm_range;
    int group;  // clamped to the backend's group limit (nss_cuda: kMaxGroup)
    // Joint multi-channel matching (nss::ssd_nch): distances are the sum over
    // `channels` planes, channel c of a frame living channel_step floats after
    // channel c - 1.
    int channels = 1;
    long long channel_step = 0;
    // Exact-size windows (nss::image_match): candidates span
    // [c - bm_range, c + range_hi] when range_hi >= 0, else the symmetric
    // [c - bm_range, c + bm_range].
    int range_hi = -1;
    NSS_GPUHOST_HD int lo() const { return bm_range > 0 ? bm_range : 0; }
    NSS_GPUHOST_HD int hi() const { return range_hi >= 0 ? range_hi : lo(); }
};

}  // namespace nss::gpuhost
