// SPDX-License-Identifier: GPL-2.0-only
// Execution slots (D28): num_streams per filter instance, each with its own
// command pools, one timeline semaphore and whatever workspace the filter
// adds. A frame records up to three stages (upload, compute, download) and
// submits them at once; the stages chain on the timeline semaphore, so the
// host waits once per frame, with a bound.
#pragma once

#include "vulkan/runtime/context.hpp"

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace nss_vk {

class Slot {
public:
    enum Stage { Upload, Compute, Download, kStages };

    // timestamps: query slots for mark() (0 = none).
    explicit Slot(std::shared_ptr<Device> device, std::uint32_t timestamps = 0);
    Slot(const Slot&) = delete;
    Slot& operator=(const Slot&) = delete;
    ~Slot();

    Device& device() const noexcept { return *device_; }
    const std::shared_ptr<Device>& shared_device() const noexcept { return device_; }

    // The stage's command buffer for the current frame, begun on first use.
    VkCommandBuffer commands(Stage stage);
    // Debug label around the commands recorded until end_label (no-op without
    // VK_EXT_debug_utils).
    void begin_label(Stage stage, const char* name);
    void end_label(Stage stage);
    // Writes timestamp `query` after the work recorded so far in `stage`.
    void mark(Stage stage, std::uint32_t query);

    // Ends and submits the recorded stages in order, each waiting for the one
    // before. A frame with nothing recorded submits nothing.
    void submit();
    // Waits for the last submitted stage (bounded: wait_timeout_ns()), then
    // resets the command pools for the next frame. Throws VkError on timeout
    // or device loss; the slot is then unusable.
    void wait();
    // Times of the marked queries in nanoseconds since query 0, after wait();
    // unmarked queries read 0. Resets the queries for the next frame.
    std::vector<double> timestamps_ns();

private:
    void release() noexcept;

    struct StageCommands {
        const Queue* queue = nullptr;
        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandBuffer buffer = VK_NULL_HANDLE;
        bool recording = false;
    };

    std::shared_ptr<Device> device_;
    StageCommands stages_[kStages];
    VkSemaphore timeline_ = VK_NULL_HANDLE;
    std::uint64_t value_ = 0;       // last value a submitted stage signals
    bool pending_ = false;          // submitted, not yet waited for
    bool broken_ = false;           // a wait failed: never reuse or free in-flight objects
    VkQueryPool queries_ = VK_NULL_HANDLE;
    std::uint32_t query_count_ = 0;
    std::vector<bool> marked_;
};

// num_streams slots per filter instance; getFrame leases one for the whole
// frame and extra VapourSynth threads wait (as nss_cuda's SlotPool).
template <class S>
class SlotPool {
public:
    explicit SlotPool(std::vector<std::unique_ptr<S>> slots) : slots_(std::move(slots)) {
        for (auto& slot : slots_) free_.push_back(slot.get());
    }
    SlotPool(const SlotPool&) = delete;
    SlotPool& operator=(const SlotPool&) = delete;

    class Lease {
    public:
        Lease(SlotPool* pool, S* slot) noexcept : pool_(pool), slot_(slot) {}
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept : pool_(other.pool_), slot_(std::exchange(other.slot_, nullptr)) {}
        ~Lease() {
            if (slot_) pool_->give_back(slot_);
        }
        S& operator*() const noexcept { return *slot_; }
        S* operator->() const noexcept { return slot_; }

    private:
        SlotPool* pool_;
        S* slot_;
    };

    Lease acquire() {
        std::unique_lock lock(mutex_);
        available_.wait(lock, [&] { return !free_.empty(); });
        S* slot = free_.back();
        free_.pop_back();
        return Lease(this, slot);
    }
    std::size_t size() const noexcept { return slots_.size(); }
    template <class F> void for_each(F&& f) {
        for (auto& slot : slots_) f(*slot);
    }

private:
    void give_back(S* slot) {
        {
            std::lock_guard lock(mutex_);
            free_.push_back(slot);
        }
        available_.notify_one();
    }
    std::vector<std::unique_ptr<S>> slots_;
    std::vector<S*> free_;
    std::mutex mutex_;
    std::condition_variable available_;
};

}  // namespace nss_vk
