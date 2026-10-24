// SPDX-License-Identifier: GPL-2.0-only
// Device block matching shared by BM3D, WNNM, MCWNNM and NCSR. Host-callable
// launchers (plain C++ header). Semantics follow the CPU matcher contract in
// src/cpu/bm/matcher.hpp: the reference patch is always entry 0; the rest is
// ordered by (finite first, distance, t, y, x); spatial search scans the
// clamped (2r+1)^2 window; temporal search is the predictive ps_num/ps_range
// layering of collect_temporal. Distances are summed in a different order
// than the CPU SIMD trees, so near-ties may resolve differently (CPU and GPU are compared by PSNR, not bitwise).
#pragma once

#include "gpuhost/geometry.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace nss_cuda {

inline constexpr int kMaxGroup = 256;  // TWSC's group limit; the others stay within nss::kBmMaxGroup (64)

struct DeviceMatch {
    int x;
    int y;
    int t;
    float dist;
};

using nss::gpuhost::MatchGeometry;
using nss::gpuhost::RasterGrid;
using nss::gpuhost::make_raster_grid;

// References [ref_begin, ref_begin + ref_count) of the grid; out holds
// ref_count * geometry.group matches (group-strided) and counts ref_count
// entries (batch-relative). With `scratch` prepared for `plane`
// (spatial_match_prepare) and a geometry spatial_match_tensor accepts, the
// tensor-core matcher runs instead of the lane kernels: the same matches, bit
// for bit.
void spatial_match(const float* plane, const MatchGeometry& geometry, const RasterGrid& grid, int ref_begin,
                   int ref_count, DeviceMatch* out, int* counts, cudaStream_t stream, const void* scratch = nullptr);

// The tensor-core matcher of large windows (sm_80+): one channel, block 8,
// group 2 to 16, a window side of 40 to 129, at least 8 references a grid
// row. Below a side of 40 the lane kernels are faster end to end (RTX 5080).
bool spatial_match_tensor(const MatchGeometry& geometry, const RasterGrid& grid);
// Bytes of its workspace for planes up to width x height.
std::size_t spatial_match_scratch_bytes(int width, int height);
// Fills the workspace from `plane` (geometry's width, height and pitch) on
// `stream`; again whenever the plane changes.
void spatial_match_prepare(const float* plane, const MatchGeometry& geometry, void* scratch, cudaStream_t stream);

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
