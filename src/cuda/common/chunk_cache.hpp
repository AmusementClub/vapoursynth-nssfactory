// SPDX-License-Identifier: GPL-2.0-only
// The rolling chunk cache of the nss_cuda nodes that roll (group_driver,
// image_filter): src/gpuhost/chunk_cache.hpp over pinned host blocks.
#pragma once

#include "cuda/runtime/memory.hpp"
#include "gpuhost/chunk_cache.hpp"

namespace nss_cuda {

using StorePool = nss::gpuhost::StorePool<PinnedBuffer>;
using ChunkStore = nss::gpuhost::ChunkStore<PinnedBuffer>;
using ChunkCache = nss::gpuhost::ChunkCache<PinnedBuffer>;
using nss::gpuhost::PlanOrder;

// ChunkCache::configure's allocator: pinned blocks charged to `budget`.
inline std::function<PinnedBuffer(std::size_t)> pinned_blocks(std::shared_ptr<nss::ResourceBudget> budget) {
    return [budget = std::move(budget)](std::size_t bytes) { return PinnedBuffer(bytes, budget); };
}

}  // namespace nss_cuda
