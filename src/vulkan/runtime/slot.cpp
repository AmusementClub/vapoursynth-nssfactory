// SPDX-License-Identifier: GPL-2.0-only
#include "vulkan/runtime/slot.hpp"

#include "vulkan/runtime/error.hpp"

#include <stdexcept>
#include <string>

namespace nss_vk {

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
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        type.initialValue = 0;
        VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type};
        NSS_VK_CHECK(vk.vkCreateSemaphore(handle, &semaphore, nullptr, &timeline_));
        if (timestamps > 0 && device_->caps().host_query_reset) {
            VkQueryPoolCreateInfo pool{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            pool.queryType = VK_QUERY_TYPE_TIMESTAMP;
            pool.queryCount = timestamps;
            NSS_VK_CHECK(vk.vkCreateQueryPool(handle, &pool, nullptr, &queries_));
            vk.vkResetQueryPool(handle, queries_, 0, timestamps);
            query_count_ = timestamps;
            marked_.assign(timestamps, false);
        }
    } catch (...) {
        release();
        throw;
    }
}

Slot::~Slot() {
    release();
}

void Slot::release() noexcept {
    const VolkDeviceTable& vk = device_->vk();
    const VkDevice handle = device_->handle();
    if (pending_ && !broken_) {
        VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        info.semaphoreCount = 1;
        info.pSemaphores = &timeline_;
        info.pValues = &value_;
        if (vk.vkWaitSemaphores(handle, &info, wait_timeout_ns()) != VK_SUCCESS) broken_ = true;
    }
    // Objects the device may still use are leaked rather than destroyed.
    if (broken_) return;
    if (queries_) vk.vkDestroyQueryPool(handle, queries_, nullptr);
    if (timeline_) vk.vkDestroySemaphore(handle, timeline_, nullptr);
    for (StageCommands& stage : stages_) {
        if (stage.pool) vk.vkDestroyCommandPool(handle, stage.pool, nullptr);
    }
    queries_ = VK_NULL_HANDLE;
    timeline_ = VK_NULL_HANDLE;
    for (StageCommands& stage : stages_) stage.pool = VK_NULL_HANDLE;
}

VkCommandBuffer Slot::commands(Stage stage) {
    if (broken_) throw std::runtime_error("nss_vk: execution slot unusable after a failed wait");
    if (pending_) throw std::logic_error("nss_vk: Slot::commands before wait()");
    StageCommands& s = stages_[stage];
    if (!s.recording) {
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        NSS_VK_CHECK(device_->vk().vkBeginCommandBuffer(s.buffer, &begin));
        s.recording = true;
    }
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
    device_->vk().vkCmdWriteTimestamp2(commands(stage), VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, queries_, query);
    marked_[query] = true;
}

void Slot::submit() {
    if (broken_) throw std::runtime_error("nss_vk: execution slot unusable after a failed wait");
    const VolkDeviceTable& vk = device_->vk();
    bool first = true;
    for (StageCommands& s : stages_) {
        if (!s.recording) continue;
        NSS_VK_CHECK(vk.vkEndCommandBuffer(s.buffer));
        s.recording = false;
        VkCommandBufferSubmitInfo buffer{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        buffer.commandBuffer = s.buffer;
        // Each stage waits for the one before it. The first waits for the
        // previous frame's last stage: the host has waited for it already, but
        // only a device-side dependency orders this frame's writes after that
        // frame's (synchronization validation: write-after-write on reused
        // buffers across queues).
        VkSemaphoreSubmitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        wait.semaphore = timeline_;
        wait.value = value_;
        wait.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        VkSemaphoreSubmitInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        signal.semaphore = timeline_;
        signal.value = value_ + 1;
        signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        VkSubmitInfo2 info{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        info.waitSemaphoreInfoCount = first && value_ == 0 ? 0u : 1u;
        info.pWaitSemaphoreInfos = &wait;
        info.commandBufferInfoCount = 1;
        info.pCommandBufferInfos = &buffer;
        info.signalSemaphoreInfoCount = 1;
        info.pSignalSemaphoreInfos = &signal;
        device_->submit(*s.queue, info);
        ++value_;
        pending_ = true;
        first = false;
    }
}

void Slot::wait() {
    if (broken_) throw std::runtime_error("nss_vk: execution slot unusable after a failed wait");
    const VolkDeviceTable& vk = device_->vk();
    const VkDevice handle = device_->handle();
    if (pending_) {
        VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        info.semaphoreCount = 1;
        info.pSemaphores = &timeline_;
        info.pValues = &value_;
        const VkResult result = vk.vkWaitSemaphores(handle, &info, wait_timeout_ns());
        if (result != VK_SUCCESS) {
            broken_ = true;
            if (result == VK_TIMEOUT) {
                throw VkError(result, "nss_vk: the device did not finish a frame within " +
                                          std::to_string(wait_timeout_ns() / 1'000'000) +
                                          " ms (NSS_VK_WAIT_TIMEOUT_MS)");
            }
            NSS_VK_CHECK(result);
            throw VkError(result, "nss_vk: vkWaitSemaphores returned " + result_name(result));
        }
        pending_ = false;
    }
    for (StageCommands& s : stages_) {
        if (s.recording) {  // recorded but never submitted: drop it
            NSS_VK_CHECK(vk.vkEndCommandBuffer(s.buffer));
            s.recording = false;
        }
        NSS_VK_CHECK(vk.vkResetCommandPool(handle, s.pool, 0));
    }
}

std::vector<double> Slot::timestamps_ns() {
    std::vector<double> times(query_count_, 0.0);
    if (query_count_ == 0 || pending_) return times;
    const VolkDeviceTable& vk = device_->vk();
    std::uint64_t origin = 0;
    std::vector<std::uint64_t> raw(query_count_, 0);
    for (std::uint32_t q = 0; q < query_count_; ++q) {
        if (!marked_[q]) continue;
        // Written and waited for: available without VK_QUERY_RESULT_WAIT_BIT,
        // which would block forever on a query that was never written.
        vk.vkGetQueryPoolResults(device_->handle(), queries_, q, 1, sizeof(std::uint64_t), &raw[q],
                                 sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT);
    }
    origin = marked_[0] ? raw[0] : 0;
    for (std::uint32_t q = 0; q < query_count_; ++q) {
        if (marked_[q]) times[q] = static_cast<double>(raw[q] - origin) * device_->timestamp_period_ns();
    }
    vk.vkResetQueryPool(device_->handle(), queries_, 0, query_count_);
    marked_.assign(query_count_, false);
    return times;
}

}  // namespace nss_vk
