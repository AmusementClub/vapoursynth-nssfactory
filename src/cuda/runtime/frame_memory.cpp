// SPDX-License-Identifier: GPL-2.0-only
#include "cuda/runtime/frame_memory.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace nss_cuda {
namespace {

void* lock(const void*, void* base, std::size_t size) {
    if (cudaHostRegister(base, size, cudaHostRegisterPortable) == cudaSuccess) return base;
    cudaGetLastError();
    return nullptr;
}

void unlock(const void*, void* base, std::size_t, void*) {
    cudaHostUnregister(base);
    cudaGetLastError();
}

const nss::frontend::FrameMemoryBackend kBackend{"NSS_CUDA_DIRECT_IO", "nss_cuda", lock, unlock};

}  // namespace

FrameMemory::FrameMemory(VSCore* core, const VSAPI* vsapi) : memory_(kBackend, core, vsapi) {}

}  // namespace nss_cuda
