// SPDX-License-Identifier: GPL-2.0-only
#include "cuda/common/nonfinite.hpp"

#include "cuda/runtime/error.hpp"

namespace nss_cuda {
namespace {

__global__ void nonfinite_kernel(const float* data, std::size_t count, int* result) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t at = i; at < count; at += stride) {
        if (!isfinite(data[at])) atomicOr(result, 1);
    }
}

}  // namespace

bool nonfinite_any(const float* data, std::size_t count, cudaStream_t stream) {
    int* device = nullptr;
    NSS_CUDA_CHECK(cudaMallocAsync(&device, sizeof(int), stream));
    NSS_CUDA_CHECK(cudaMemsetAsync(device, 0, sizeof(int), stream));
    constexpr int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    nonfinite_kernel<<<blocks, threads, 0, stream>>>(data, count, device);
    NSS_CUDA_CHECK_LAUNCH();
    int result = 0;
    NSS_CUDA_CHECK(cudaMemcpyAsync(&result, device, sizeof(result), cudaMemcpyDeviceToHost, stream));
    NSS_CUDA_CHECK(cudaStreamSynchronize(stream));
    NSS_CUDA_CHECK(cudaFreeAsync(device, stream));
    return result != 0;
}

}  // namespace nss_cuda
