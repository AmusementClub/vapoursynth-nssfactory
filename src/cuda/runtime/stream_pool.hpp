// SPDX-License-Identifier: GPL-2.0-only
// num_streams execution slots per filter instance. getFrame
// leases a slot for the whole frame; extra VapourSynth threads wait instead of
// creating unbounded streams/buffers.
#pragma once

#include "cuda/runtime/error.hpp"

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <utility>
#include <vector>

namespace nss_cuda {

// Flags of an event a host thread waits for (cudaEventSynchronize): the
// thread sleeps until it is signalled. Without cudaEventBlockingSync it spins
// in the driver, whatever the context's scheduling flags (which belong to the
// process's primary context, shared with other plugins): the spinning threads
// cost up to 48x the CPU per frame at the same speed
// (docs/api/cuda/README.md#host-waits).
inline constexpr unsigned kHostWaitEvent = cudaEventDisableTiming | cudaEventBlockingSync;
// Flags of an event only streams wait for (cudaStreamWaitEvent). Recording a
// blocking event costs more and stalls at times (rolling BM3D YUV420:
// cudaEventRecord mean 1.0 -> 49.5 us, p99 3.7 -> 703 us, 1.4% fewer fps), so
// these stay without it.
inline constexpr unsigned kStreamEvent = cudaEventDisableTiming;

class Stream {
public:
    Stream() {
        NSS_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        const cudaError_t error = cudaEventCreateWithFlags(&done_, kHostWaitEvent);
        if (error != cudaSuccess) {
            cudaStreamDestroy(stream_);  // the destructor does not run for a constructor that throws
            NSS_CUDA_CHECK(error);
        }
    }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    ~Stream() {
        if (done_) cudaEventDestroy(done_);
        if (stream_) cudaStreamDestroy(stream_);
    }
    cudaStream_t get() const noexcept { return stream_; }
    operator cudaStream_t() const noexcept { return stream_; }
    // Waits for the work queued so far, sleeping (cudaStreamSynchronize
    // would spin: see kHostWaitEvent). One thread uses a stream at a time.
    void synchronize() const {
        NSS_CUDA_CHECK(cudaEventRecord(done_, stream_));
        NSS_CUDA_CHECK(cudaEventSynchronize(done_));
    }

private:
    cudaStream_t stream_ = nullptr;
    cudaEvent_t done_ = nullptr;
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
        // The slot free the longest: frames that give their slot back once
        // their work is queued then rotate over the streams, so that one
        // frame's copies overlap another's kernels. Taking the slot just given
        // back (LIFO) put every frame on one stream: num_streams had no effect.
        Slot* slot = free_.front();
        free_.pop_front();
        return Lease(this, slot);
    }
    // As acquire(), but the waiting thread with the lowest order goes first
    // (a frame number: requests that VapourSynth's threads bring in any order
    // then take their turns in clip order). A pool is used with one of the two.
    Lease acquire(long long order) {
        std::unique_lock lock(mutex_);
        const auto me = waiting_.insert(order);
        available_.wait(lock, [&] { return !free_.empty() && *waiting_.begin() == order; });
        waiting_.erase(me);
        Slot* slot = free_.front();
        free_.pop_front();
        const bool more = !free_.empty() && !waiting_.empty();
        lock.unlock();
        if (more) available_.notify_all();
        return Lease(this, slot);
    }
    std::size_t size() const noexcept { return slots_.size(); }
    template <class F> void for_each(F&& f) {
        for (auto& slot : slots_) f(*slot);
    }

private:
    void give_back(Slot* slot) {
        bool ordered;
        {
            std::lock_guard lock(mutex_);
            free_.push_back(slot);
            ordered = !waiting_.empty();
        }
        // An ordered waiter must be the lowest one: wake them all.
        if (ordered) {
            available_.notify_all();
        } else {
            available_.notify_one();
        }
    }
    std::vector<std::unique_ptr<Slot>> slots_;
    std::deque<Slot*> free_;
    std::multiset<long long> waiting_;  // orders of the threads in acquire(order)
    std::mutex mutex_;
    std::condition_variable available_;
};

}  // namespace nss_cuda
