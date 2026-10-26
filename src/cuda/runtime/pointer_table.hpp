// SPDX-License-Identifier: GPL-2.0-only
// Device tables of plane pointers written on a stream by a kernel that takes
// the values as its arguments. The CUDA driver copies kernel arguments when
// the launch is made (cuLaunchKernel: each parameter "will be copied" from the
// memory kernelParams points to), so no host buffer has to outlive the call.
// A cudaMemcpyAsync from pageable memory promises neither that nor asynchrony:
// for it the runtime only says the copy "might be synchronous with respect to
// host" and "may synchronize with the stream" (CUDA Runtime API, "API
// synchronization behavior").
#pragma once

#include <cuda_runtime.h>

namespace nss_cuda {

inline constexpr int kMaxTablePointers = 128;

// table[0, n) = values[0, n), ordered on `stream` with the work before and
// after it. 0 <= n <= kMaxTablePointers.
void write_pointer_table(void* table, const void* const* values, int n, cudaStream_t stream);

}  // namespace nss_cuda
