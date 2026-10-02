// SPDX-License-Identifier: GPL-2.0-only
// num_streams execution slots per filter instance (plan.md §6.3). getFrame
// leases a slot for the whole frame; extra VapourSynth threads wait instead of
// creating unbounded streams/buffers.
#pragma once

#include "cuda/runtime/error.hpp"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace nss_cuda {

class Stream {
public:
    Stream() { NSS_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    ~Stream() {
        if (stream_) cudaStreamDestroy(stream_);
    }
    cudaStream_t get() const noexcept { return stream_; }
    operator cudaStream_t() const noexcept { return stream_; }
    void synchronize() const { NSS_CUDA_CHECK(cudaStreamSynchronize(stream_)); }

private:
    cudaStream_t stream_ = nullptr;
};

// Slot owns whatever a frame needs (stream, arena, staging); built at creation.
template <class Slot>
class SlotPool {
public:
    explicit SlotPool(std::vector<std::unique_ptr<Slot>> slots) : slots_(std::move(slots)) {
        for (auto& slot : slots_) free_.push_back(slot.get());
    }
    SlotPool(const SlotPool&) = delete;
    SlotPool& operator=(const SlotPool&) = delete;

    class Lease {
    public:
        Lease(SlotPool* pool, Slot* slot) noexcept : pool_(pool), slot_(slot) {}
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept : pool_(other.pool_), slot_(std::exchange(other.slot_, nullptr)) {}
        ~Lease() {
            if (slot_) pool_->give_back(slot_);
        }
        Slot& operator*() const noexcept { return *slot_; }
        Slot* operator->() const noexcept { return slot_; }

    private:
        SlotPool* pool_;
        Slot* slot_;
    };

    Lease acquire() {
        std::unique_lock lock(mutex_);
        available_.wait(lock, [&] { return !free_.empty(); });
        Slot* slot = free_.back();
        free_.pop_back();
        return Lease(this, slot);
    }
    std::size_t size() const noexcept { return slots_.size(); }
    template <class F> void for_each(F&& f) {
        for (auto& slot : slots_) f(*slot);
    }

private:
    void give_back(Slot* slot) {
        {
            std::lock_guard lock(mutex_);
            free_.push_back(slot);
        }
        available_.notify_one();
    }
    std::vector<std::unique_ptr<Slot>> slots_;
    std::vector<Slot*> free_;
    std::mutex mutex_;
    std::condition_variable available_;
};

}  // namespace nss_cuda
