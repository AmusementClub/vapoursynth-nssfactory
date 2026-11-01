// SPDX-License-Identifier: GPL-2.0-only
// The rolling chunk cache of the nss_cuda nodes that roll (group_driver,
// image_filter): src/gpuhost/chunk_cache.hpp over host or device blocks.
#pragma once

#include "cuda/runtime/memory.hpp"
#include "gpuhost/chunk_cache.hpp"

namespace nss_cuda {

class ChunkBuffer {
public:
    ChunkBuffer() = default;
    ChunkBuffer(std::size_t bytes, const std::shared_ptr<nss::ResourceBudget>& budget, bool device)
        : device_(device ? DeviceBuffer(bytes, budget) : DeviceBuffer()),
          host_(device ? PinnedBuffer() : PinnedBuffer(bytes, budget)), device_resident_(device) {}
    ChunkBuffer(ChunkBuffer&&) noexcept = default;
    ChunkBuffer& operator=(ChunkBuffer&&) noexcept = default;
    ChunkBuffer(const ChunkBuffer&) = delete;
    ChunkBuffer& operator=(const ChunkBuffer&) = delete;

    void* get() const noexcept { return device_resident_ ? device_.get() : host_.get(); }
    template <class T> T* as() const { return static_cast<T*>(get()); }
    bool device_resident() const noexcept { return device_resident_; }

private:
    DeviceBuffer device_;
    PinnedBuffer host_;
    bool device_resident_ = false;
};

using StorePool = nss::gpuhost::StorePool<ChunkBuffer>;
using ChunkStore = nss::gpuhost::ChunkStore<ChunkBuffer>;
using ChunkCache = nss::gpuhost::ChunkCache<ChunkBuffer>;
using nss::gpuhost::PlanOrder;

inline std::function<ChunkBuffer(std::size_t)> chunk_blocks(std::shared_ptr<nss::ResourceBudget> budget, bool device) {
    return [budget = std::move(budget), device](std::size_t bytes) { return ChunkBuffer(bytes, budget, device); };
}

}  // namespace nss_cuda
