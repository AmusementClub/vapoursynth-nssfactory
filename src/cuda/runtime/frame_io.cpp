// SPDX-License-Identifier: GPL-2.0-only
#include "cuda/runtime/frame_io.hpp"
#include "cuda/runtime/error.hpp"

#include <cstdint>
#include <cstring>

namespace nss_cuda {
namespace {

void copy_rows(const void* src, std::ptrdiff_t src_stride, void* dst, std::ptrdiff_t dst_stride, std::size_t row_bytes,
               int rows) {
    const auto* in = static_cast<const std::uint8_t*>(src);
    auto* out = static_cast<std::uint8_t*>(dst);
    if (src_stride == dst_stride && static_cast<std::size_t>(src_stride) == row_bytes) {
        std::memcpy(out, in, row_bytes * static_cast<std::size_t>(rows));
        return;
    }
    for (int row = 0; row < rows; ++row) {
        std::memcpy(out + row * dst_stride, in + row * src_stride, row_bytes);
    }
}

}  // namespace

void upload_plane(const void* host, std::ptrdiff_t host_stride, std::size_t row_bytes, int rows, void* staging,
                  void* device, std::size_t device_pitch, cudaStream_t stream) {
    copy_rows(host, host_stride, staging, static_cast<std::ptrdiff_t>(row_bytes), row_bytes, rows);
    NSS_CUDA_CHECK(cudaMemcpy2DAsync(device, device_pitch, staging, row_bytes, row_bytes, static_cast<std::size_t>(rows),
                                     cudaMemcpyHostToDevice, stream));
}

void begin_download(const void* device, std::size_t device_pitch, std::size_t row_bytes, int rows, void* staging,
                    cudaStream_t stream) {
    NSS_CUDA_CHECK(cudaMemcpy2DAsync(staging, row_bytes, device, device_pitch, row_bytes, static_cast<std::size_t>(rows),
                                     cudaMemcpyDeviceToHost, stream));
}

void finish_download(const void* staging, std::size_t row_bytes, int rows, void* host, std::ptrdiff_t host_stride) {
    copy_rows(staging, static_cast<std::ptrdiff_t>(row_bytes), host, host_stride, row_bytes, rows);
}

}  // namespace nss_cuda
