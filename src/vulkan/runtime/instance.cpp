// SPDX-License-Identifier: GPL-2.0-only
#include "vulkan/runtime/instance.hpp"

#include "vulkan/runtime/error.hpp"

namespace nss_vk {
namespace {

Instance create_instance() {
    Instance result;
    if (volkInitialize() != VK_SUCCESS) {
#if defined(_WIN32)
        result.error = "no Vulkan loader (vulkan-1.dll not found)";
#elif defined(__APPLE__)
        result.error = "no Vulkan loader (libvulkan.dylib not found)";
#else
        result.error = "no Vulkan loader (libvulkan.so.1 not found)";
#endif
        return result;
    }
    result.loader_version = volkGetInstanceVersion();
    if (result.loader_version < kRequiredApiVersion) {
        result.error = "Vulkan loader " + version_text(result.loader_version) + " is older than 1.3";
        return result;
    }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "nss_vk";
    app.pEngineName = "nssfactory";
    app.apiVersion = kRequiredApiVersion;
    VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    info.pApplicationInfo = &app;
    VkInstance handle = VK_NULL_HANDLE;
    if (const VkResult code = vkCreateInstance(&info, nullptr, &handle); code != VK_SUCCESS) {
        result.error = "vkCreateInstance failed: " + result_name(code);
        return result;
    }
    // Instance-level entry points only; every device gets its own table.
    volkLoadInstanceOnly(handle);
    std::uint32_t count = 0;
    VkResult code = vkEnumeratePhysicalDevices(handle, &count, nullptr);
    if (code == VK_SUCCESS) {
        result.devices.resize(count);
        code = vkEnumeratePhysicalDevices(handle, &count, result.devices.data());
        result.devices.resize(count);
    }
    if (code < VK_SUCCESS) {
        result.devices.clear();
        result.error = "vkEnumeratePhysicalDevices failed: " + result_name(code);
    } else if (result.devices.empty()) {
        result.error = "no Vulkan device";
    }
    result.handle = handle;
    return result;
}

}  // namespace

const Instance& instance() {
    static const Instance shared = create_instance();
    return shared;
}

std::string version_text(std::uint32_t version) {
    if (version == 0) {
        return "none";
    }
    return std::to_string(VK_API_VERSION_MAJOR(version)) + "." + std::to_string(VK_API_VERSION_MINOR(version)) + "." +
           std::to_string(VK_API_VERSION_PATCH(version));
}

}  // namespace nss_vk
