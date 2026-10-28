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

// The square matcher (match_square.cu; Davy & Ehret 2021, section 4.1.4):
// blocks of twice the step (2 at 1, 4 at 2, 8 at 4, 12 at 6, 16 at 8), one
// channel, groups 2 to 32, symmetric windows. Quarter distances are shared by
// the four references that contain them, so the distances differ from the
// lane kernels' in the last bits (near ties may resolve differently); keys
// and order are the same. spatial_match_square_shape: the geometries it can
// run; spatial_match_square_takes: those where it is faster than the lane
// kernels. Returns false (and does nothing) for a geometry the latter
// rejects, or a batch of fewer thread blocks than the device has SMs (a few
// grid rows: the lane kernels are faster then); with `any_shape` (tests,
// timing) only for one the former rejects. Against the
// tensor-core matcher it wins up to a window radius of 30 (1.15-1.9x) and
// ties at 40: callers leave radii of kSquareTensorRadius and more to it where
// it applies.
inline constexpr int kSquareTensorRadius = 40;
bool spatial_match_square_shape(const MatchGeometry& geometry, const RasterGrid& grid);
bool spatial_match_square_takes(const MatchGeometry& geometry, const RasterGrid& grid);
bool spatial_match_square(const float* plane, const MatchGeometry& geometry, const RasterGrid& grid, int ref_begin,
                          int ref_count, DeviceMatch* out, int* counts, cudaStream_t stream, bool any_shape = false);
// The same search of frame `frame` of the device pointer table `frames`
// (read on the device, in stream order).
bool spatial_match_square_frame(const float* const* frames, int frame, const MatchGeometry& geometry,
                                const RasterGrid& grid, int ref_begin, int ref_count, DeviceMatch* out, int* counts,
                                cudaStream_t stream, bool any_shape = false);

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

// The predictive search with its centre frame by the square matcher: the
// centre window's matches as spatial_match_square finds them (the same keys
// up to near ties), then the temporal layers of predictive_match seeded from
// them. Returns false (and does nothing) where the square matcher declines
// the centre frame (spatial_match_square_takes, or a batch too small to fill
// the device; with `any_shape` only spatial_match_square_shape) or the lane
// kernels do not serve the geometry; predictive_match is the search then.
bool predictive_match_square(const MatchGeometry& geometry, const TemporalWindow& window, const RasterGrid& grid,
                             int ref_begin, int ref_count, DeviceMatch* out, int* counts, cudaStream_t stream,
                             bool any_shape = false);

}  // namespace nss_cuda
