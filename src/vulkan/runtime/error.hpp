// SPDX-License-Identifier: GPL-2.0-only
// Vulkan error handling: every call that returns a VkResult goes through
// NSS_VK_CHECK, which throws VkError; VapourSynth boundaries turn the exception
// into a filter error, never a silent fallback (as nss_cuda's CudaError).
#pragma once

#include "vulkan/runtime/vk.hpp"

#include <stdexcept>
#include <string>

namespace nss_vk {

class VkError : public std::runtime_error {
public:
    VkError(VkResult code, const std::string& message) : std::runtime_error(message), code_(code) {}
    VkResult code() const noexcept { return code_; }

private:
    VkResult code_;
};

// "VK_ERROR_DEVICE_LOST" etc.; unknown codes fall back to the number.
std::string result_name(VkResult code);

[[noreturn]] void throw_vk_error(VkResult code, const char* expression, const char* file, int line);

inline void check_vk(VkResult code, const char* expression, const char* file, int line) {
    if (code < VK_SUCCESS) {
        throw_vk_error(code, expression, file, line);
    }
}

}  // namespace nss_vk

// Negative results are errors; positive ones (VK_TIMEOUT, VK_INCOMPLETE, ...)
// are returned to the caller, which must handle them explicitly.
#define NSS_VK_CHECK(expression) ::nss_vk::check_vk((expression), #expression, __FILE__, __LINE__)
