// SPDX-License-Identifier: GPL-2.0-only
// Device block matching shared by BM3D, WNNM, MCWNNM and NCSR. Host-callable
// launchers (plain C++ header). Semantics follow the CPU matcher contract in
// src/cpu/bm/matcher.hpp: the reference patch is always entry 0; the rest is
// ordered by (finite first, distance, t, y, x); spatial search scans the
// clamped (2r+1)^2 window; temporal search is the predictive ps_num/ps_range
// layering of collect_temporal. Distances are summed in a different order
// than the CPU SIMD trees, so near-ties may resolve differently (D9/D16).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace nss_cuda {

inline constexpr int kMaxGroup = 64;  // == nss::kBmMaxGroup

struct DeviceMatch {
    int x;
    int y;
    int t;
    float dist;
};

// Reference positions: the CPU raster grid (append_raster_jobs) with the last
// row/column clamped to (width - block, height - block).
struct RasterGrid {
    int nx;
    int ny;
    int step;
    int max_x;
    int max_y;
    __host__ __device__ int count() const { return nx * ny; }
    __host__ __device__ int x(int i) const { const int v = (i % nx) * step; return v < max_x ? v : max_x; }
    __host__ __device__ int y(int i) const { const int v = (i / nx) * step; return v < max_y ? v : max_y; }
};
RasterGrid make_raster_grid(int width, int height, int block, int step);

struct MatchGeometry {
    int width;
    int height;
    int pitch;  // in floats
    int block;
    int bm_range;
    int group;  // clamped to kMaxGroup
    // Joint multi-channel matching (nss::ssd_nch): distances are the sum over
    // `channels` planes, channel c of a frame living channel_step floats after
    // channel c - 1.
    int channels = 1;
    long long channel_step = 0;
    // Exact-size windows (nss::image_match): candidates span
    // [c - bm_range, c + range_hi] when range_hi >= 0, else the symmetric
    // [c - bm_range, c + bm_range].
    int range_hi = -1;
    __host__ __device__ int lo() const { return bm_range > 0 ? bm_range : 0; }
    __host__ __device__ int hi() const { return range_hi >= 0 ? range_hi : lo(); }
};

// References [ref_begin, ref_begin + ref_count) of the grid; out holds
// ref_count * geometry.group matches (group-strided) and counts ref_count
// entries (batch-relative).
void spatial_match(const float* plane, const MatchGeometry& geometry, const RasterGrid& grid, int ref_begin,
                   int ref_count, DeviceMatch* out, int* counts, cudaStream_t stream);

struct TemporalWindow {
    const float* const* frames;  // device array of ntemp device plane pointers (same pitch)
    int ntemp;
    int t0;
    int radius;
    int valid_begin;  // first usable index (clip boundary), see SearchConfig::valid_t_begin
    int valid_end;    // one past the last usable index
    int ps_num;
    int ps_range;
};

void predictive_match(const MatchGeometry& geometry, const TemporalWindow& window, const RasterGrid& grid,
                      int ref_begin, int ref_count, DeviceMatch* out, int* counts, cudaStream_t stream);

}  // namespace nss_cuda
