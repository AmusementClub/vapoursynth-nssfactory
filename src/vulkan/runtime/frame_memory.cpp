// SPDX-License-Identifier: GPL-2.0-only
#include "vulkan/runtime/frame_memory.hpp"

#include <atomic>
#include <cstdint>

namespace nss_vk {
namespace {

constexpr VkExternalMemoryHandleTypeFlagBits kHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;

struct Import {
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
};

// Imports in the process: each is an allocation of its own, and devices
// allow maxMemoryAllocationCount (4096 on NVIDIA) for everything.
std::atomic<std::uint32_t> imports{0};

void* lock(const void* domain, void* base, std::size_t size) {
    const Device& device = *static_cast<const Device*>(domain);
    const VolkDeviceTable& vk = device.vk();
    const VkDevice handle = device.handle();
    const std::uint64_t align = device.caps().import_alignment;
    if (align == 0 || reinterpret_cast<std::uintptr_t>(base) % align != 0 || size % align != 0) return nullptr;
    if (imports.fetch_add(1) >= device.caps().max_allocations / 4) {
        imports.fetch_sub(1);
        return nullptr;
    }
    VkMemoryHostPointerPropertiesEXT pointer{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    // A coherent type: the copies' writes reach the host with the download
    // stage's host barrier and the slot's wait, and the host's writes reach
    // the copies with the submission (the import is never mapped, so nothing
    // could be flushed or invalidated).
    std::uint32_t type = UINT32_MAX;
    if (vk.vkGetMemoryHostPointerPropertiesEXT(handle, kHandleType, base, &pointer) == VK_SUCCESS) {
        VkPhysicalDeviceMemoryProperties memory{};
        vkGetPhysicalDeviceMemoryProperties(device.physical(), &memory);
        constexpr VkMemoryPropertyFlags wanted = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (std::uint32_t t = 0; t < memory.memoryTypeCount && type == UINT32_MAX; ++t) {
            if ((pointer.memoryTypeBits >> t & 1u) && (memory.memoryTypes[t].propertyFlags & wanted) == wanted) type = t;
        }
    }
    auto* import = new Import;
    const auto fail = [&]() -> void* {
        if (import->buffer) vk.vkDestroyBuffer(handle, import->buffer, nullptr);
        if (import->memory) vk.vkFreeMemory(handle, import->memory, nullptr);
        delete import;
        imports.fetch_sub(1);
        return nullptr;
    };
    if (type == UINT32_MAX) return fail();

    VkExternalMemoryBufferCreateInfo external{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    external.handleTypes = kHandleType;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &external};
    info.size = size;
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    // The copy queues may be other families than compute (as Buffer).
    const std::vector<std::uint32_t>& families = device.families();
    if (families.size() > 1) {
        info.sharingMode = VK_SHARING_MODE_CONCURRENT;
        info.queueFamilyIndexCount = static_cast<std::uint32_t>(families.size());
        info.pQueueFamilyIndices = families.data();
    }
    if (vk.vkCreateBuffer(handle, &info, nullptr, &import->buffer) != VK_SUCCESS) {
        import->buffer = VK_NULL_HANDLE;
        return fail();
    }
    VkMemoryRequirements requirements{};
    vk.vkGetBufferMemoryRequirements(handle, import->buffer, &requirements);
    if (!(requirements.memoryTypeBits >> type & 1u) || requirements.size > size) return fail();

    VkImportMemoryHostPointerInfoEXT host{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
    host.handleType = kHandleType;
    host.pHostPointer = base;
    VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &host};
    allocate.allocationSize = size;
    allocate.memoryTypeIndex = type;
    if (vk.vkAllocateMemory(handle, &allocate, nullptr, &import->memory) != VK_SUCCESS) {
        import->memory = VK_NULL_HANDLE;
        return fail();
    }
    if (vk.vkBindBufferMemory(handle, import->buffer, import->memory, 0) != VK_SUCCESS) return fail();
    return import;
}

void unlock(const void* domain, void*, std::size_t, void* handle) {
    const Device& device = *static_cast<const Device*>(domain);
    auto* import = static_cast<Import*>(handle);
    // A poisoned device may still run copies on it: leak it (as Buffer).
    if (!device.poisoned()) {
        device.vk().vkDestroyBuffer(device.handle(), import->buffer, nullptr);
        device.vk().vkFreeMemory(device.handle(), import->memory, nullptr);
    }
    delete import;
    imports.fetch_sub(1);
}

const nss::frontend::FrameMemoryBackend kBackend{"NSS_VK_DIRECT_IO", "nss_vk", lock, unlock};

}  // namespace

FrameMemory::FrameMemory(std::shared_ptr<Device> device, VSCore* core, const VSAPI* vsapi) : device_(std::move(device)) {
    if (device_->host_import()) memory_.emplace(kBackend, core, vsapi, device_.get());
}

HostPlane FrameMemory::pinned(const VSFrame* f, int plane, const void* data, std::ptrdiff_t stride,
                              std::size_t row_bytes) {
    if (!memory_ || stride != static_cast<std::ptrdiff_t>(row_bytes)) return {};
    const nss::frontend::PinnedPlane lock = memory_->pinned(f, plane, data);
    if (!lock) return {};
    return {static_cast<const Import*>(lock.handle)->buffer, lock.offset};
}

}  // namespace nss_vk
