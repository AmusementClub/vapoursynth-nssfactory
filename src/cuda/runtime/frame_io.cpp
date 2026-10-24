// SPDX-License-Identifier: GPL-2.0-only
#include "cuda/runtime/frame_io.hpp"
#include "cuda/runtime/error.hpp"

#include <cstdint>
#include <cstring>
#if defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#define NSS_CUDA_STREAM_STORES 1
#else
#define NSS_CUDA_STREAM_STORES 0
#endif

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

// Rows into a VapourSynth frame with non-temporal stores: the destination is
// usually not in cache, and plain stores read every line before writing it.
// Only the rolling outputs' copies from the chunk store use them (BM3D
// radius 1 RGB 1080p 1.07x). finish_download copies: there they made the
// copy itself faster (805 to 651 us a 1080p plane under NCSR) but the next
// upload slower (331 to 515 us); at the RGB 1080p defaults in continuous
// playback NCSR ran 1.27-1.30x and BM3D up to 1.16x faster without them
// (docs/cuda-rebench-20261020.md). Other targets copy.
void stream_rows(const void* src, std::ptrdiff_t src_stride, void* dst, std::ptrdiff_t dst_stride, std::size_t row_bytes,
                 int rows) {
#if NSS_CUDA_STREAM_STORES
    const auto row = [](const std::uint8_t* in, std::uint8_t* out, std::size_t bytes) {
        std::size_t head = (16 - (reinterpret_cast<std::uintptr_t>(out) & 15)) & 15;
        if (head > bytes) head = bytes;
        std::memcpy(out, in, head);
        std::size_t i = head;
        for (; i + 64 <= bytes; i += 64) {
            const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + i));
            const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + i + 16));
            const __m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + i + 32));
            const __m128i d = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + i + 48));
            _mm_stream_si128(reinterpret_cast<__m128i*>(out + i), a);
            _mm_stream_si128(reinterpret_cast<__m128i*>(out + i + 16), b);
            _mm_stream_si128(reinterpret_cast<__m128i*>(out + i + 32), c);
            _mm_stream_si128(reinterpret_cast<__m128i*>(out + i + 48), d);
        }
        for (; i + 16 <= bytes; i += 16) {
            _mm_stream_si128(reinterpret_cast<__m128i*>(out + i), _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + i)));
        }
        std::memcpy(out + i, in + i, bytes - i);
    };
    const auto* in = static_cast<const std::uint8_t*>(src);
    auto* out = static_cast<std::uint8_t*>(dst);
    if (src_stride == dst_stride && static_cast<std::size_t>(src_stride) == row_bytes) {
        row(in, out, row_bytes * static_cast<std::size_t>(rows));
    } else {
        for (int r = 0; r < rows; ++r) row(in + r * src_stride, out + r * dst_stride, row_bytes);
    }
    _mm_sfence();  // the stores are visible before the frame is handed on
#else
    copy_rows(src, src_stride, dst, dst_stride, row_bytes, rows);
#endif
}

}  // namespace

void upload_plane(const void* host, std::ptrdiff_t host_stride, std::size_t row_bytes, int rows, void* staging,
                  void* device, std::size_t device_pitch, cudaStream_t stream) {
    stage_plane(host, host_stride, row_bytes, rows, staging);
    upload_staged(staging, row_bytes, rows, device, device_pitch, stream);
}

void stage_plane(const void* host, std::ptrdiff_t host_stride, std::size_t row_bytes, int rows, void* staging) {
    copy_rows(host, host_stride, staging, static_cast<std::ptrdiff_t>(row_bytes), row_bytes, rows);
}

void upload_staged(const void* staging, std::size_t row_bytes, int rows, void* device, std::size_t device_pitch,
                   cudaStream_t stream) {
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

void upload_direct(const void* host, std::ptrdiff_t host_stride, std::size_t row_bytes, int rows, void* device,
                   std::size_t device_pitch, cudaStream_t stream) {
    NSS_CUDA_CHECK(cudaMemcpy2DAsync(device, device_pitch, host, static_cast<std::size_t>(host_stride), row_bytes,
                                     static_cast<std::size_t>(rows), cudaMemcpyHostToDevice, stream));
}

void download_direct(const void* device, std::size_t device_pitch, std::size_t row_bytes, int rows, void* host,
                     std::ptrdiff_t host_stride, cudaStream_t stream) {
    NSS_CUDA_CHECK(cudaMemcpy2DAsync(host, static_cast<std::size_t>(host_stride), device, device_pitch, row_bytes,
                                     static_cast<std::size_t>(rows), cudaMemcpyDeviceToHost, stream));
}

void copy_to_frame(const void* src, std::ptrdiff_t src_stride, void* dst, std::ptrdiff_t dst_stride, std::size_t row_bytes,
                   int rows) {
    stream_rows(src, src_stride, dst, dst_stride, row_bytes, rows);
}

}  // namespace nss_cuda
