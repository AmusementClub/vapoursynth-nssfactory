// SPDX-License-Identifier: GPL-2.0-only
#include "vulkan/runtime/slot.hpp"

#include "vulkan/runtime/error.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#if NSS_VK_GPU_FRAMES
#include <VSVulkan4.h>
#endif

namespace nss_vk {

bool timing_enabled() {
    const char* value = std::getenv("NSS_VK_TIMES");
    return value && *value && std::strcmp(value, "0") != 0;
}

Slot::Slot(std::shared_ptr<Device> device, std::uint32_t timestamps) : device_(std::move(device)) {
    const VolkDeviceTable& vk = device_->vk();
    const VkDevice handle = device_->handle();
    const Queue* queues[kStages] = {&device_->upload(), &device_->compute(), &device_->download()};
    try {
        for (int s = 0; s < kStages; ++s) {
            StageCommands& stage = stages_[s];
            stage.queue = queues[s];
            VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            pool.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
            pool.queueFamilyIndex = stage.queue->family;
            NSS_VK_CHECK(vk.vkCreateCommandPool(handle, &pool, nullptr, &stage.pool));
            VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            allocate.commandPool = stage.pool;
            allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocate.commandBufferCount = 1;
            NSS_VK_CHECK(vk.vkAllocateCommandBuffers(handle, &allocate, &stage.buffer));
        }
        if (device_->core_device()) {
#if NSS_VK_GPU_FRAMES
            // A timeline of the core's (initial value 0): the planes it is
            // published on hold references of their own.
            char error[256] = {};
            gpu_timeline_ = device_->core_api()->createGPUTimeline(device_->core(), error, sizeof(error));
            if (!gpu_timeline_) throw std::runtime_error(std::string("nss_vk: createGPUTimeline failed: ") + error);
            timeline_ = device_->core_api()->getGPUTimelineSemaphore(gpu_timeline_);
#endif
        } else {
            VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
            type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
            type.initialValue = 0;
            VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type};
            NSS_VK_CHECK(vk.vkCreateSemaphore(handle, &semaphore, nullptr, &timeline_));
        }
        if (timestamps > 0 && device_->caps().host_query_reset) {
            VkQueryPoolCreateInfo pool{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            pool.queryType = VK_QUERY_TYPE_TIMESTAMP;
            pool.queryCount = timestamps;
            NSS_VK_CHECK(vk.vkCreateQueryPool(handle, &pool, nullptr, &queries_));
            vk.vkResetQueryPool(handle, queries_, 0, timestamps);
            query_count_ = timestamps;
            marked_.assign(timestamps, -1);
            times_.assign(timestamps, 0.0);
        }
    } catch (...) {
        release();
        throw;
    }
}

Slot::~Slot() {
    settle();
    release_frames();  // nothing after a failed wait (see retain)
    release();
}

void Slot::retain(const VSFrame* f, const VSAPI* vsapi) {
    retained_.reserve(retained_.size() + 1);  // may throw: before the reference is taken
    retained_.emplace_back(vsapi->addFrameRef(f), vsapi);
}

void Slot::release_frames() noexcept {
    // A broken slot's work may still run: its frames stay. The device is then
    // poisoned, but another slot's work that its wait saw finish is done, so
    // that slot's frames go (later frames would leak them all otherwise).
    if (state_ == State::broken) return;
    for (const auto& [frame, vsapi] : retained_) vsapi->freeFrame(frame);
    retained_.clear();
}

void Slot::release() noexcept {
    // Objects the device may still use are leaked rather than destroyed: this
    // slot's, and on a poisoned device any (a broken slot's batch may still
    // wait on this slot's timeline).
    if (state_ == State::broken || state_ == State::pending || device_->poisoned()) return;
    const VolkDeviceTable& vk = device_->vk();
    const VkDevice handle = device_->handle();
    if (queries_) vk.vkDestroyQueryPool(handle, queries_, nullptr);
#if NSS_VK_GPU_FRAMES
    if (gpu_timeline_) {
        device_->core_api()->freeGPUTimeline(gpu_timeline_);
    } else
#endif
    if (timeline_) {
        vk.vkDestroySemaphore(handle, timeline_, nullptr);
    }
    gpu_timeline_ = nullptr;
    for (StageCommands& stage : stages_) {
        if (stage.pool) vk.vkDestroyCommandPool(handle, stage.pool, nullptr);
        stage.pool = VK_NULL_HANDLE;
    }
    queries_ = VK_NULL_HANDLE;
    timeline_ = VK_NULL_HANDLE;
}

void Slot::fail() noexcept {
    state_ = State::broken;
    device_->poison();
}

VkResult Slot::wait_timeline() noexcept {
    VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    info.semaphoreCount = 1;
    info.pSemaphores = &timeline_;
    info.pValues = &value_;
    return device_->vk().vkWaitSemaphores(device_->handle(), &info, wait_timeout_ns());
}

void Slot::reset_commands() noexcept {
    const VolkDeviceTable& vk = device_->vk();
    for (StageCommands& s : stages_) {
        if (s.recording) {  // recorded but never submitted: drop it
            vk.vkEndCommandBuffer(s.buffer);
            s.recording = false;
        }
        if (s.pool) vk.vkResetCommandPool(device_->handle(), s.pool, 0);
        s.waits.clear();
        s.done = 0;
    }
}

VkCommandBuffer Slot::commands(Stage stage) {
    if (state_ == State::broken) throw std::runtime_error("nss_vk: execution slot unusable after a failed wait");
    if (state_ == State::pending) throw std::logic_error("nss_vk: Slot::commands before wait()");
    StageCommands& s = stages_[stage];
    if (!s.recording) {
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        NSS_VK_CHECK(device_->vk().vkBeginCommandBuffer(s.buffer, &begin));
        s.recording = true;
    }
    state_ = State::recording;
    return s.buffer;
}

void Slot::begin_label(Stage stage, const char* name) {
    if (!device_->debug_labels()) return;
    VkDebugUtilsLabelEXT label{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
    label.pLabelName = name;
    vkCmdBeginDebugUtilsLabelEXT(commands(stage), &label);
}

void Slot::end_label(Stage stage) {
    if (!device_->debug_labels()) return;
    vkCmdEndDebugUtilsLabelEXT(commands(stage));
}

void Slot::mark(Stage stage, std::uint32_t query) {
    if (query >= query_count_ || stages_[stage].queue->timestamp_bits == 0) return;
    if (marked_[query] >= 0) throw std::logic_error("nss_vk: timestamp query marked twice in one frame");
    device_->vk().vkCmdWriteTimestamp2(commands(stage), VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, queries_, query);
    marked_[query] = stage;
}

void Slot::wait_for(Stage stage, VkSemaphore semaphore, std::uint64_t value) {
    if (semaphore == VK_NULL_HANDLE || semaphore == timeline_) return;
    commands(stage);  // the stage must be submitted for the wait to apply
    for (auto& wait : stages_[stage].waits) {
        if (wait.first == semaphore) {
            wait.second = std::max(wait.second, value);
            return;
        }
    }
    stages_[stage].waits.emplace_back(semaphore, value);
}

std::pair<VkSemaphore, std::uint64_t> Slot::done(Stage stage) const noexcept {
    const std::uint64_t value = stages_[stage].done;
    return {value ? timeline_ : VK_NULL_HANDLE, value};
}

void Slot::submit_stage(StageCommands& s) {
    NSS_VK_CHECK(device_->vk().vkEndCommandBuffer(s.buffer));
    s.recording = false;
    VkCommandBufferSubmitInfo buffer{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    buffer.commandBuffer = s.buffer;
    // Each stage waits for the one before it. The first of a frame waits for
    // the previous frame's last stage: the host has waited for it already, but
    // only a device-side dependency orders this frame's writes after that
    // frame's (synchronization validation: write-after-write on reused buffers
    // across queues).
    std::vector<VkSemaphoreSubmitInfo> waits;
    if (value_ != 0) {
        VkSemaphoreSubmitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        wait.semaphore = timeline_;
        wait.value = value_;
        wait.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        waits.push_back(wait);
    }
    for (const auto& [semaphore, value] : s.waits) {
        VkSemaphoreSubmitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        wait.semaphore = semaphore;
        wait.value = value;
        wait.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        waits.push_back(wait);
    }
    VkSemaphoreSubmitInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    signal.semaphore = timeline_;
    signal.value = value_ + 1;
    signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkSubmitInfo2 info{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    info.waitSemaphoreInfoCount = static_cast<std::uint32_t>(waits.size());
    info.pWaitSemaphoreInfos = waits.data();
    info.commandBufferInfoCount = 1;
    info.pCommandBufferInfos = &buffer;
    info.signalSemaphoreInfoCount = 1;
    info.pSignalSemaphoreInfos = &signal;
    device_->submit(*s.queue, info);
    s.done = ++value_;
    ++submit_count_;
}

void Slot::submit() {
    if (state_ == State::broken) throw std::runtime_error("nss_vk: execution slot unusable after a failed wait");
    if (state_ != State::recording) return;
    submit_count_ = 0;
    // With one copy queue for both directions, a download batch would wait
    // there for its kernel and hold every later upload (queues run their
    // batches in order): it is submitted by wait() once the kernel is done, so
    // that the uploads of other slots run ahead (D28 as amended).
    const bool defer = device_->defer_downloads() && stages_[Download].recording &&
                       (stages_[Upload].recording || stages_[Compute].recording);
    bool submitted = false;
    for (int stage = 0; stage < kStages; ++stage) {
        StageCommands& s = stages_[stage];
        if (!s.recording || (defer && stage == Download)) continue;
        // Stages submitted before a failing one stay pending; settle() waits
        // for them (value_ names the last one that was submitted).
        try {
            submit_stage(s);
        } catch (...) {
            state_ = submitted ? State::pending : State::recording;
            throw;
        }
        submitted = true;
    }
    deferred_ = defer;
    state_ = State::pending;
}

void Slot::wait() {
    if (state_ == State::broken) throw std::runtime_error("nss_vk: execution slot unusable after a failed wait");
    if (state_ == State::pending) {
        wait_or_throw();
        if (deferred_) {
            deferred_ = false;
            submit_stage(stages_[Download]);
            wait_or_throw();
        }
        read_timestamps();
    }
    reset_commands();
    state_ = State::idle;
    release_frames();
}

void Slot::wait_or_throw() {
    const VkResult result = wait_timeline();
    if (result == VK_SUCCESS) return;
    fail();
    if (result == VK_TIMEOUT) {
        throw VkError(result, "nss_vk: the device did not finish a frame within " +
                                  std::to_string(wait_timeout_ns() / 1'000'000) + " ms (NSS_VK_WAIT_TIMEOUT_MS)");
    }
    throw VkError(result, "nss_vk: waiting for a frame failed: " + result_name(result));
}

void Slot::settle() noexcept {
    if (state_ == State::broken) return;
    if (state_ == State::idle && query_count_ == 0) {  // wait() has reset the commands
        release_frames();
        return;
    }
    if (state_ == State::pending && wait_timeline() != VK_SUCCESS) {
        fail();
        return;
    }
    deferred_ = false;  // a download never submitted is dropped with the commands
    reset_commands();
    if (query_count_) {
        device_->vk().vkResetQueryPool(device_->handle(), queries_, 0, query_count_);
        marked_.assign(query_count_, -1);
    }
    state_ = State::idle;
    release_frames();
}

void Slot::read_timestamps() noexcept {
    if (query_count_ == 0) return;
    const VolkDeviceTable& vk = device_->vk();
    times_.assign(query_count_, 0.0);
    std::vector<std::uint64_t> raw(query_count_, 0);
    std::vector<bool> valid(query_count_, false);
    for (std::uint32_t q = 0; q < query_count_; ++q) {
        if (marked_[q] < 0) continue;
        // Written and waited for: available without VK_QUERY_RESULT_WAIT_BIT,
        // which would block forever on a query that was never written.
        if (vk.vkGetQueryPoolResults(device_->handle(), queries_, q, 1, sizeof(std::uint64_t), &raw[q],
                                     sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) {
            continue;
        }
        const std::uint32_t bits = stages_[marked_[q]].queue->timestamp_bits;
        if (bits < 64) raw[q] &= (std::uint64_t{1} << bits) - 1;
        valid[q] = true;
    }
    if (valid[0]) {
        for (std::uint32_t q = 0; q < query_count_; ++q) {
            if (valid[q] && raw[q] >= raw[0]) {
                times_[q] = static_cast<double>(raw[q] - raw[0]) * device_->timestamp_period_ns();
            }
        }
    }
    // Host reset after the wait: every query is unavailable again before the
    // next frame writes it.
    vk.vkResetQueryPool(device_->handle(), queries_, 0, query_count_);
    marked_.assign(query_count_, -1);
}

}  // namespace nss_vk
