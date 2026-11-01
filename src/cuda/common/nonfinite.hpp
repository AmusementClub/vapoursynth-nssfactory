// SPDX-License-Identifier: GPL-2.0-only
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace nss_cuda {

bool nonfinite_any(const float* data, std::size_t count, cudaStream_t stream);

}  // namespace nss_cuda
