// SPDX-License-Identifier: GPL-2.0-only
// NVTX ranges for Nsight Systems (header-only NVTX3; no cost without a tool).
#pragma once

#include <nvtx3/nvToolsExt.h>

namespace nss_cuda {

class NvtxRange {
public:
    explicit NvtxRange(const char* name) noexcept { nvtxRangePushA(name); }
    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
    ~NvtxRange() { nvtxRangePop(); }
};

}  // namespace nss_cuda

#define NSS_CUDA_CONCAT_INNER(a, b) a##b
#define NSS_CUDA_CONCAT(a, b) NSS_CUDA_CONCAT_INNER(a, b)
#define NSS_CUDA_RANGE(name) ::nss_cuda::NvtxRange NSS_CUDA_CONCAT(nss_cuda_range_, __LINE__)(name)
