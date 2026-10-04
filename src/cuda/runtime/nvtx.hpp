// SPDX-License-Identifier: GPL-2.0-only
// NVTX ranges for Nsight Systems (header-only NVTX3; no cost without a tool).
#pragma once

// Minimal toolkit installs may omit the NVTX headers; ranges then compile away.
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define NSS_CUDA_HAVE_NVTX 1
#else
#define NSS_CUDA_HAVE_NVTX 0
#endif

namespace nss_cuda {

class NvtxRange {
public:
#if NSS_CUDA_HAVE_NVTX
    explicit NvtxRange(const char* name) noexcept { nvtxRangePushA(name); }
    ~NvtxRange() { nvtxRangePop(); }
#else
    explicit NvtxRange(const char*) noexcept {}
    ~NvtxRange() {}
#endif
    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

}  // namespace nss_cuda

#define NSS_CUDA_CONCAT_INNER(a, b) a##b
#define NSS_CUDA_CONCAT(a, b) NSS_CUDA_CONCAT_INNER(a, b)
#define NSS_CUDA_RANGE(name) ::nss_cuda::NvtxRange NSS_CUDA_CONCAT(nss_cuda_range_, __LINE__)(name)
