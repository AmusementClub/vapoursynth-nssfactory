// SPDX-License-Identifier: GPL-2.0-only
// SPIR-V embedded at build time (nss_vk_add_shader in cmake/Vulkan.cmake): one
// entry per variant id listed in src/vulkan/CMakeLists.txt.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace nss_vk {

struct Spirv {
    const std::uint32_t* words = nullptr;
    std::size_t count = 0;  // words
};

// Throws std::logic_error for an unknown id (a build-list mistake).
Spirv shader(std::string_view id);
// "glslang vulkan-sdk-..., SPIRV-Tools vulkan-sdk-..." or "system".
const char* shader_tools();
// FNV-1a over every embedded variant (ids and words): names the on-disk
// pipeline cache of this build.
std::uint64_t shader_set_hash();

}  // namespace nss_vk
