// SPDX-License-Identifier: GPL-2.0-only
// VapourSynth plane <-> device transfers through pinned staging (D12: frames
// enter and leave as CPU VSFrames). Each call needs its own staging region
// until the stream has finished with it.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace nss_cuda {

// Host rows (any stride) -> staging (tight) -> device (pitched), async on stream.
void upload_plane(const void* host, std::ptrdiff_t host_stride, std::size_t row_bytes, int rows, void* staging,
                  void* device, std::size_t device_pitch, cudaStream_t stream);

// Device (pitched) -> staging (tight), async on stream. After the stream has
// synchronized, finish_download copies staging into the host rows.
void begin_download(const void* device, std::size_t device_pitch, std::size_t row_bytes, int rows, void* staging,
                    cudaStream_t stream);
void finish_download(const void* staging, std::size_t row_bytes, int rows, void* host, std::ptrdiff_t host_stride);

}  // namespace nss_cuda
