// SPDX-License-Identifier: GPL-2.0-only
// Image-load probe: a trivial kernel whose first launch forces the driver to
// pick the device's cubin or JIT-compile the embedded PTX (D13).
#include "cuda/runtime/device.hpp"

#include <cuda_runtime.h>

#include <chrono>

namespace nss_cuda {
namespace {

constexpr int kProbeValue = 0x4e5353;  // "NSS"

__global__ void probe_kernel(int* out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *out = kProbeValue;
    }
}

std::string message(const char* step, cudaError_t error) {
    return std::string(step) + ": " + cudaGetErrorName(error) + ": " + cudaGetErrorString(error);
}

}  // namespace

ProbeResult probe(int device) {
    ProbeResult result;
    if (const cudaError_t error = cudaSetDevice(device); error != cudaSuccess) {
        result.error = message("cudaSetDevice", error);
        return result;
    }
    int* value = nullptr;
    if (const cudaError_t error = cudaMalloc(&value, sizeof(int)); error != cudaSuccess) {
        result.error = message("cudaMalloc", error);
        return result;
    }
    const auto start = std::chrono::steady_clock::now();
    probe_kernel<<<1, 32>>>(value);
    cudaError_t error = cudaGetLastError();
    int host = 0;
    if (error == cudaSuccess) {
        error = cudaMemcpy(&host, value, sizeof(int), cudaMemcpyDeviceToHost);
    }
    result.milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    cudaFree(value);
    if (error != cudaSuccess) {
        result.error = message("probe launch", error);
    } else if (host != kProbeValue) {
        result.error = "probe kernel returned a wrong value";
    } else {
        result.ok = true;
    }
    return result;
}

}  // namespace nss_cuda
