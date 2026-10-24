// SPDX-License-Identifier: GPL-2.0-only
// Finished rolling chunks on the host, shared by the GPU nodes that roll
// (nss_cuda's group_driver and image_filter, nss_vk's to come; D27): chunk
// stores of host memory, reused through a pool, and an LRU cache of finished
// chunks that grows when chunks it dropped are asked for again
// (rolling_cache_chunks / rolling_cache_limit). Backend-free: `Bytes` is the
// backend's host block (nss_cuda: PinnedBuffer), with get() null when empty
// and as<T>() its mapped bytes.
#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <set>
#include <vector>

namespace nss::gpuhost {

// Host blocks for chunk stores, reused so that a chunk neither allocates nor
// faults in fresh pages. A block goes back when its chunk is dropped.
template <class Bytes>
struct StorePool {
    std::mutex mu;
    std::vector<Bytes> free;
};

// The finished frames of one chunk. The device downloads straight into it.
template <class Bytes>
struct ChunkStore {
    int start = 0, count = 0;
    Bytes bytes;  // rolling_chunk frames, each its planes back to back (tight rows)
    std::size_t frame_bytes = 0;
    std::array<std::size_t, 3> plane_offset{};
    std::shared_ptr<StorePool<Bytes>> pool;
    // Final plane of frame start + i.
    float* plane(int i, int p) const {
        return reinterpret_cast<float*>(bytes.template as<std::uint8_t>() + i * frame_bytes + plane_offset[p]);
    }
    ChunkStore() = default;
    ChunkStore(const ChunkStore&) = delete;
    ChunkStore& operator=(const ChunkStore&) = delete;
    ~ChunkStore() {
        if (!pool || !bytes.get()) return;
        std::lock_guard lock(pool->mu);
        pool->free.push_back(std::move(bytes));
    }
};

template <class Bytes>
class ChunkCache {
public:
    using Store = ChunkStore<Bytes>;
    // Every store holds `chunk` frames of `frame_bytes` (planes at plane_offset).
    // The cache starts at `size` chunks and may grow to `cap`. `allocate`
    // makes a block of the given size (charged to the filter's budget).
    void configure(int chunk, std::size_t frame_bytes, const std::array<std::size_t, 3>& plane_offset, int size, int cap,
                   std::function<Bytes(std::size_t)> allocate) {
        chunk_ = chunk;
        frame_bytes_ = frame_bytes;
        plane_offset_ = plane_offset;
        size_ = size;
        cap_ = cap;
        allocate_ = std::move(allocate);
    }

    // An empty store for the chunk [start, start + count), from the pool when it has a block.
    std::shared_ptr<Store> make_store(int start, int count) {
        auto store = std::make_shared<Store>();
        store->start = start;
        store->count = count;
        store->frame_bytes = frame_bytes_;
        store->plane_offset = plane_offset_;
        {
            std::lock_guard lock(mu_);  // clear() replaces the pool
            store->pool = stores_;
        }
        {
            std::lock_guard lock(store->pool->mu);
            if (!store->pool->free.empty()) {
                store->bytes = std::move(store->pool->free.back());
                store->pool->free.pop_back();
            }
        }
        if (!store->bytes.get()) store->bytes = allocate_(frame_bytes_ * chunk_);
        return store;
    }

    // The chunk starting at `start`: from the cache, from the thread already
    // computing it, or from compute() (which returns a shared_ptr<Store>).
    template <class Compute>
    std::shared_ptr<const Store> get(int start, Compute&& compute) {
        std::unique_lock lock(mu_);
        for (;;) {
            for (auto it = cache_.begin(); it != cache_.end(); ++it) {
                if ((*it)->start == start) {
                    auto hit = *it;
                    cache_.splice(cache_.begin(), cache_, it);
                    return hit;
                }
            }
            if (!computing_.count(start)) break;
            cv_.wait(lock);  // another thread is computing this chunk
        }
        // A miss on a chunk the cache dropped recently: after two of them the
        // cache keeps one more chunk, up to cap.
        if (size_ < cap_) {
            const auto again = std::find(dropped_.begin(), dropped_.end(), start);
            if (again != dropped_.end()) {
                dropped_.erase(again);
                if (++dropped_hits_ >= 2) {
                    dropped_hits_ = 0;
                    ++size_;
                }
            }
        }
        computing_.insert(start);
        lock.unlock();
        std::shared_ptr<const Store> computed;
        try {
            computed = compute();
        } catch (...) {
            lock.lock();
            computing_.erase(start);
            cv_.notify_all();
            throw;
        }
        lock.lock();
        computing_.erase(start);
        cache_.push_front(computed);
        while (static_cast<int>(cache_.size()) > size_) {
            dropped_.push_back(cache_.back()->start);
            cache_.pop_back();
        }
        while (dropped_.size() > static_cast<std::size_t>(std::max(8, 2 * cap_))) dropped_.pop_front();
        cv_.notify_all();
        return computed;
    }

    // A thread is computing the chunk starting at `start`.
    bool computing(int start) {
        std::lock_guard lock(mu_);
        return computing_.count(start) != 0;
    }

    // Frees the stores (CUDA: call with the device current).
    void clear() {
        std::lock_guard lock(mu_);
        cache_.clear();
        stores_ = std::make_shared<StorePool<Bytes>>();
    }

private:
    int chunk_ = 1, size_ = 1, cap_ = 1;
    std::size_t frame_bytes_ = 0;
    std::array<std::size_t, 3> plane_offset_{};
    std::function<Bytes(std::size_t)> allocate_;
    std::shared_ptr<StorePool<Bytes>> stores_ = std::make_shared<StorePool<Bytes>>();
    std::deque<int> dropped_;  // starts of recently dropped chunks
    int dropped_hits_ = 0;
    std::mutex mu_;
    std::condition_variable cv_;
    std::list<std::shared_ptr<const Store>> cache_;
    std::set<int> computing_;
};

// Rolling chunks carry on from the chunk planned before them on a stream,
// which saves the centers before the chunk's start, but frame requests reach
// the nodes out of order (1080p, 16 threads: half of MCWNNM's chunks started
// afresh). A chunk therefore waits, before it takes a staging set and plans,
// until the chunk before it is planned: as long as that one is being
// computed (it plans first thing), and for kWait while it has not been asked
// for (sequential requests arrive within that). It holds nothing meanwhile.
class PlanOrder {
public:
    static constexpr auto kWait = std::chrono::milliseconds(5);

    template <class Cache>
    void wait_for_previous(int start, int chunk, Cache& cache) {
        const int previous = start - chunk;
        if (previous < 0) return;
        std::unique_lock lock(mu_);
        const auto since = std::chrono::steady_clock::now();
        while (!planned_.count(previous) &&
               (cache.computing(previous) || std::chrono::steady_clock::now() - since < kWait)) {
            cv_.wait_for(lock, std::chrono::milliseconds(1));
        }
    }
    void planned(int start, int chunk) {
        {
            std::lock_guard lock(mu_);
            planned_.insert(start);
            // Only the recent past matters to a chunk that waits.
            planned_.erase(planned_.begin(), planned_.lower_bound(start - 64 * chunk));
        }
        cv_.notify_all();
    }

private:
    std::mutex mu_;
    std::condition_variable cv_;
    std::set<int> planned_;
};

}  // namespace nss::gpuhost
