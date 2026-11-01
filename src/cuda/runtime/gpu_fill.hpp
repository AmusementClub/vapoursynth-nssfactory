// SPDX-License-Identifier: GPL-2.0-only
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace nss_cuda {

void gpu_fill(void* target, std::size_t pitch, int rows, std::uint32_t word, cudaStream_t stream);

}  // namespace nss_cuda
