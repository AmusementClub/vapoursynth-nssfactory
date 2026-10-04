// SPDX-License-Identifier: GPL-2.0-only
#include "cuda/runtime/device.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace nss_cuda {
namespace {

// Leading 0 keeps the arrays well-formed when a list is empty; it never
// matches a real architecture.
constexpr int kCubinArchs[] = {0, NSS_CUDA_CUBIN_ARCHS};
constexpr int kPtxArchs[] = {0, NSS_CUDA_PTX_ARCHS};

std::string cuda_message(cudaError_t error) {
    return std::string(cudaGetErrorName(error)) + ": " + cudaGetErrorString(error);
}

}  // namespace

const char* support_name(Support support) {
    switch (support) {
    case Support::native:
        return "native";
    case Support::jit:
        return "jit";
    default:
        return "unsupported";
    }
}

Support classify(int cc_major, int cc_minor, bool ptx_jit_available) {
    const int device = cc_major * 10 + cc_minor;
    for (const int arch : kCubinArchs) {
        if (arch != 0 && arch / 10 == cc_major && arch % 10 <= cc_minor) {
            return Support::native;
        }
    }
    for (const int arch : kPtxArchs) {
        if (ptx_jit_available && arch != 0 && arch <= device) {
            return Support::jit;
        }
    }
    return Support::unsupported;
}

const char* compiled_architectures() {
    return NSS_CUDA_ARCH_STRING;
}

RuntimeInfo runtime_info() {
    RuntimeInfo info;
    cudaRuntimeGetVersion(&info.runtime_version);
    if (const cudaError_t error = cudaDriverGetVersion(&info.driver_version); error != cudaSuccess) {
        info.error = cuda_message(error);
        return info;
    }
    if (const cudaError_t error = cudaGetDeviceCount(&info.device_count); error != cudaSuccess) {
        info.device_count = 0;
        info.error = cuda_message(error);
    }
    return info;
}

DeviceInfo device_info(int device, const RuntimeInfo& runtime) {
    int count = 0;
    if (const cudaError_t error = cudaGetDeviceCount(&count); error != cudaSuccess) {
        throw std::runtime_error("no usable CUDA driver/device (" + cuda_message(error) + ")");
    }
    if (device < 0 || device >= count) {
        throw std::runtime_error("device_id " + std::to_string(device) + " is out of range (" +
                                 std::to_string(count) + " CUDA device(s))");
    }
    cudaDeviceProp prop{};
    if (const cudaError_t error = cudaGetDeviceProperties(&prop, device); error != cudaSuccess) {
        throw std::runtime_error("cannot query CUDA device " + std::to_string(device) + " (" + cuda_message(error) + ")");
    }
    DeviceInfo info;
    info.index = device;
    info.name = prop.name;
    info.cc_major = prop.major;
    info.cc_minor = prop.minor;
    info.total_memory = prop.totalGlobalMem;
    info.multiprocessors = prop.multiProcessorCount;
    info.support = classify(prop.major, prop.minor, runtime.ptx_jit_available());
    return info;
}

}  // namespace nss_cuda
