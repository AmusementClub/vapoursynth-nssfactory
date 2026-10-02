// SPDX-License-Identifier: GPL-2.0-only
// CUDA error handling: every runtime call goes through NSS_CUDA_CHECK, which
// throws CudaError; VapourSynth boundaries (checked_frame / creation) turn the
// exception into a filter error, never a silent fallback.
#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace nss_cuda {

class CudaError : public std::runtime_error {
public:
    CudaError(cudaError_t code, const std::string& message) : std::runtime_error(message), code_(code) {}
    cudaError_t code() const noexcept { return code_; }

private:
    cudaError_t code_;
};

[[noreturn]] void throw_cuda_error(cudaError_t code, const char* expression, const char* file, int line);

inline void check_cuda(cudaError_t code, const char* expression, const char* file, int line) {
    if (code != cudaSuccess) {
        throw_cuda_error(code, expression, file, line);
    }
}

}  // namespace nss_cuda

#define NSS_CUDA_CHECK(expression) ::nss_cuda::check_cuda((expression), #expression, __FILE__, __LINE__)
// After a kernel launch: reports launch-configuration errors immediately.
#define NSS_CUDA_CHECK_LAUNCH() ::nss_cuda::check_cuda(cudaGetLastError(), "kernel launch", __FILE__, __LINE__)
