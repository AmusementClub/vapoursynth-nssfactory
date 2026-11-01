// SPDX-License-Identifier: GPL-2.0-only
#include "cuda/runtime/gpu_fill.hpp"

#include "cuda/runtime/error.hpp"

namespace nss_cuda {
namespace {

__global__ void fill_kernel(std::uint8_t* target, std::size_t pitch, int rows, std::uint32_t word) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    const std::size_t count = pitch * static_cast<std::size_t>(rows) / sizeof(word);
    for (std::size_t at = i; at < count; at += stride) {
        const std::size_t row = at / (pitch / sizeof(word));
        const std::size_t column = at % (pitch / sizeof(word));
        *reinterpret_cast<std::uint32_t*>(target + row * pitch + column * sizeof(word)) = word;
    }
}

}  // namespace

void gpu_fill(void* target, std::size_t pitch, int rows, std::uint32_t word, cudaStream_t stream) {
    const std::size_t count = pitch * static_cast<std::size_t>(rows) / sizeof(word);
    constexpr int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    fill_kernel<<<blocks, threads, 0, stream>>>(static_cast<std::uint8_t*>(target), pitch, rows, word);
    NSS_CUDA_CHECK_LAUNCH();
}

}  // namespace nss_cuda
