// SPDX-License-Identifier: GPL-2.0-only
#include "cuda/runtime/pointer_table.hpp"
#include "cuda/runtime/error.hpp"

#include <stdexcept>

namespace nss_cuda {
namespace {

struct Values {
    const void* p[kMaxTablePointers];
};

__global__ void pointer_table_kernel(const void** table, Values values, int n) {
    const int i = static_cast<int>(threadIdx.x);
    if (i < n) table[i] = values.p[i];
}

}  // namespace

void write_pointer_table(void* table, const void* const* values, int n, cudaStream_t stream) {
    if (n < 0 || n > kMaxTablePointers) throw std::logic_error("nss_cuda: pointer table larger than kMaxTablePointers");
    if (n == 0) return;
    Values v{};
    for (int i = 0; i < n; ++i) v.p[i] = values[i];
    pointer_table_kernel<<<1, kMaxTablePointers, 0, stream>>>(static_cast<const void**>(table), v, n);
    NSS_CUDA_CHECK_LAUNCH();
}

}  // namespace nss_cuda
