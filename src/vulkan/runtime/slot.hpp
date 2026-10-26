// SPDX-License-Identifier: GPL-2.0-only
// Execution slots (D28): num_streams per filter instance, each with its own
// command pools, one timeline semaphore and whatever workspace the filter
// adds. A frame records up to three stages (upload, compute, download) and
// submits them at once; the stages chain on the timeline semaphore, so the
// host waits once per frame, with a bound.
//
// Frame protocol: commands() / mark() / labels, then submit(), then wait().
// Host writes into upload buffers must happen before submit(). settle()
// returns the slot to idle from any state (the SlotPool calls it whenever a
// lease ends, also on error paths), so no frame is left recording or pending.
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
    // Debug label around commands of one stage (no-op without
    // VK_EXT_debug_utils); begin and end in the same stage.
    void begin_label(Stage stage, const char* name);
    void end_label(Stage stage);
    // Writes timestamp `query` (once per frame) after the work recorded so far
    // in `stage`. Times are comparable only between queries of one stage's
    // queue (copy and compute queues have unrelated clocks in general).
    void mark(Stage stage, std::uint32_t query);

    // Makes `stage` of this frame wait until another slot's timeline reaches
    // `value` (a frame that slot uploads into a cache shared by the slots).
    void wait_for(Stage stage, VkSemaphore semaphore, std::uint64_t value);
    // After submit(): the timeline and value that mark `stage` of this frame
    // done, for other slots' wait_for. {null, 0} when the stage was empty.
    std::pair<VkSemaphore, std::uint64_t> done(Stage stage) const noexcept;

    // Ends and submits the recorded stages in order, each waiting for the one
    // before. A frame with nothing recorded submits nothing. On a device with
    // one copy queue for both directions the download stage is submitted by
    // wait() once the frame's kernels are done.
    void submit();
    // Waits for the submitted stages (bounded: wait_timeout_ns()) and readies
    // the slot for the next frame. Throws VkError on timeout or device loss;
    // the slot is then unusable and the device is poisoned (Device::poison).
    void wait();
    // Times of this frame's marked queries in nanoseconds since query 0 (0 for
    // unmarked or unavailable ones). Valid after wait() until the next frame.
    const std::vector<double>& timestamps_ns() const noexcept { return times_; }

    // Back to idle from any state, never throwing: waits for a pending frame
    // (bounded), drops commands that were recorded but not submitted.
    void settle() noexcept;

    // Keeps f referenced until this frame's submitted work is done: the work
    // reads or writes its memory (frames imported for direct I/O). The
    // references go once wait() or settle() saw the work finish. After this
    // slot's wait failed they are leaked with its objects, so that
    // VapourSynth never reuses memory the device may still access.
    void retain(const VSFrame* f, const VSAPI* vsapi);

private:
    enum class State { idle, recording, pending, broken };
    struct StageCommands {
        const Queue* queue = nullptr;
        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandBuffer buffer = VK_NULL_HANDLE;
        bool recording = false;
        std::vector<std::pair<VkSemaphore, std::uint64_t>> waits;  // other slots' timelines
        std::uint64_t done = 0;  // the value its submission signals, 0: not submitted
    };

    void release() noexcept;
    void release_frames() noexcept;
    void submit_stage(StageCommands& stage);
    void wait_or_throw();
    VkResult wait_timeline() noexcept;
    void reset_commands() noexcept;
    void read_timestamps() noexcept;
    void fail() noexcept;

    std::shared_ptr<Device> device_;
    StageCommands stages_[kStages];
    VkSemaphore timeline_ = VK_NULL_HANDLE;
    std::uint64_t value_ = 0;  // last value a submitted stage signals
    State state_ = State::idle;
    bool deferred_ = false;  // the download stage waits for wait() (Device::defer_downloads)
    VkQueryPool queries_ = VK_NULL_HANDLE;
    std::uint32_t query_count_ = 0;
    std::vector<int> marked_;  // per query: the stage that wrote it, -1 = none
    std::vector<double> times_;
    std::vector<std::pair<const VSFrame*, const VSAPI*>> retained_;
};

// num_streams slots per filter instance; getFrame leases one for the whole
// frame and extra VapourSynth threads wait (as nss_cuda's SlotPool). `S` is a
// Slot or a filter struct holding one; it provides settle(), which the pool
// calls when a lease ends so that a frame abandoned by an exception never
// leaks into the next one.
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
        slot->settle();
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
