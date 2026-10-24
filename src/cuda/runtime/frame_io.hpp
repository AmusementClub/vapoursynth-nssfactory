// SPDX-License-Identifier: GPL-2.0-only
// VapourSynth plane <-> device transfers through pinned staging (frames
// enter and leave as CPU VSFrames). Each call needs its own staging region
// until the stream has finished with it.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace nss_cuda {

// Host rows (any stride) -> staging (tight) -> device (pitched), async on stream.
void upload_plane(const void* host, std::ptrdiff_t host_stride, std::size_t row_bytes, int rows, void* staging,
                  void* device, std::size_t device_pitch, cudaStream_t stream);

// The two halves of upload_plane, for callers that fill staging before they
// hold a stream: host rows -> staging, then staging -> device on the stream.
void stage_plane(const void* host, std::ptrdiff_t host_stride, std::size_t row_bytes, int rows, void* staging);
void upload_staged(const void* staging, std::size_t row_bytes, int rows, void* device, std::size_t device_pitch,
                   cudaStream_t stream);

// Device (pitched) -> staging (tight), async on stream. After the stream has
// synchronized, finish_download copies staging into the host rows.
void begin_download(const void* device, std::size_t device_pitch, std::size_t row_bytes, int rows, void* staging,
                    cudaStream_t stream);
void finish_download(const void* staging, std::size_t row_bytes, int rows, void* host, std::ptrdiff_t host_stride);
// Direct transfers between page-locked host rows (a VapourSynth frame plane
// that FrameMemory::pinned() accepted) and the device, async on stream: no
// staging. The host rows must stay allocated until the stream has finished.
void upload_direct(const void* host, std::ptrdiff_t host_stride, std::size_t row_bytes, int rows, void* device,
                   std::size_t device_pitch, cudaStream_t stream);
void download_direct(const void* device, std::size_t device_pitch, std::size_t row_bytes, int rows, void* host,
                     std::ptrdiff_t host_stride, cudaStream_t stream);
// Host rows into a frame being output, with non-temporal stores where the
// target has them (the rolling outputs; finish_download copies plainly).
void copy_to_frame(const void* src, std::ptrdiff_t src_stride, void* dst, std::ptrdiff_t dst_stride, std::size_t row_bytes,
                   int rows);

}  // namespace nss_cuda
