// SPDX-License-Identifier: GPL-2.0-only
// nss_cuda.BM4D device pipeline: the CPU model (include/nss/params/bm4d.hpp)
// ported operation for operation (distances in the CPU's order, the same
// transform arithmetic, the fixed-point sums), so a volume's output is the
// CPU's bit for bit.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace nss_cuda {

// Device work bytes for a volume of t frames of h x w.
std::size_t bm4d_volume_work_bytes(int t, int h, int w);
// The constant tables (DCT matrices and Kaiser windows of both cube sizes);
// host holds bm4d_table_floats() floats.
std::size_t bm4d_table_floats();
void bm4d_fill_tables(float* host);
// Both stages on one contiguous volume (device), into out (device).
void bm4d_run_volume(const float* noisy, int t, int h, int w, float sigma, float* out, const float* tables,
                     void* work, std::size_t work_bytes, cudaStream_t stream);
// dst (t x h x w, contiguous) = rows [y0, y0 + h) and columns [x0, x0 + w) of
// the frames (pitch in floats), frame z of the volume from frames[z].
void bm4d_gather(const float* const* frames, std::size_t pitch, int t, int y0, int x0, int h, int w, float* dst,
                 cudaStream_t stream);
// The inverse for the kept part: rows [ky0, ky1), columns [kx0, kx1) of
// volume frames [z0, z0 + count) into the frames (pitch in floats), whose
// origins are (y0, x0) in plane coordinates.
void bm4d_scatter(const float* src, int h, int w, int y0, int x0, int z0, int count, int ky0, int ky1, int kx0, int kx1,
                  float* const* frames, std::size_t pitch, cudaStream_t stream);

}  // namespace nss_cuda
