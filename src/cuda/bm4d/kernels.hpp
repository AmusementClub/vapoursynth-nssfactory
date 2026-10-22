// SPDX-License-Identifier: GPL-2.0-only
// nss_cuda.BM4D device pipeline: the CPU model (include/nss/params/bm4d.hpp)
// with the same transform arithmetic and fixed-point sums; the matching takes
// FP16 tensor-core distances on sm_80+ (the CPU's exact distances elsewhere),
// so the output is the CPU's to its PSNR, not bit for bit. Every run gives the
// same bits.
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
// The tables into constant memory on the current device (host: the filled
// tables); before bm4d_run_volume.
void bm4d_upload_constants(const float* host);
// The part [z0, z1) x [y0, y1) x [x0, x1) of a volume whose output is used.
struct Bm4dKeep {
    int z0, z1, y0, y1, x0, x1;
};
// Both stages on one contiguous volume (device), into out (device); out is
// the CPU's within keep only (the Wiener stage skips the groups that have no
// cube there).
void bm4d_run_volume(const float* noisy, int t, int h, int w, float sigma, const Bm4dKeep& keep, float* out, void* work,
                     std::size_t work_bytes, cudaStream_t stream);
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
