// SPDX-License-Identifier: GPL-2.0-only
#include "cuda/runtime/memory.hpp"

#include <string>

namespace nss_cuda {

void throw_cuda_error(cudaError_t code, const char* expression, const char* file, int line) {
    // Clear a non-sticky error so the next call on this thread is not blamed.
    cudaGetLastError();
    std::string where(file);
    const auto slash = where.find_last_of("/\\");
    if (slash != std::string::npos) where.erase(0, slash + 1);
    throw CudaError(code, std::string("nss_cuda: ") + cudaGetErrorName(code) + " (" + cudaGetErrorString(code) +
                              ") in " + expression + " at " + where + ":" + std::to_string(line));
}

DeviceBuffer::DeviceBuffer(std::size_t bytes, const std::shared_ptr<nss::ResourceBudget>& budget)
    : charge_(budget, nss::ResourceKind::Workspace, bytes) {
    if (bytes) {
        NSS_CUDA_CHECK(cudaMalloc(&ptr_, bytes));
        bytes_ = bytes;
    }
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        reset();
        ptr_ = std::exchange(other.ptr_, nullptr);
        bytes_ = std::exchange(other.bytes_, 0);
        charge_ = std::move(other.charge_);
    }
    return *this;
}

void DeviceBuffer::reset() noexcept {
    if (ptr_) cudaFree(ptr_);
    ptr_ = nullptr;
    bytes_ = 0;
    charge_ = BudgetCharge();
}

PinnedBuffer::PinnedBuffer(std::size_t bytes, const std::shared_ptr<nss::ResourceBudget>& budget)
    : charge_(budget, nss::ResourceKind::Pinned, bytes) {
    if (bytes) {
        NSS_CUDA_CHECK(cudaMallocHost(&ptr_, bytes));
        bytes_ = bytes;
    }
}

PinnedBuffer& PinnedBuffer::operator=(PinnedBuffer&& other) noexcept {
    if (this != &other) {
        reset();
        ptr_ = std::exchange(other.ptr_, nullptr);
        bytes_ = std::exchange(other.bytes_, 0);
        charge_ = std::move(other.charge_);
    }
    return *this;
}

void PinnedBuffer::reset() noexcept {
    if (ptr_) cudaFreeHost(ptr_);
    ptr_ = nullptr;
    bytes_ = 0;
    charge_ = BudgetCharge();
}

}  // namespace nss_cuda
