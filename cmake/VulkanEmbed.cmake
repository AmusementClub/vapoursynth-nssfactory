# SPDX-License-Identifier: GPL-2.0-only
# cmake -DNSS_VK_ID=<id> -DNSS_VK_SPV=<in.spv> -DNSS_VK_CPP=<out.cpp> -P VulkanEmbed.cmake
# Writes the SPIR-V words of one shader variant as a C++ array (see
# nss_vk_add_shader in Vulkan.cmake and src/vulkan/runtime/shaders.hpp).
file(READ "${NSS_VK_SPV}" hex HEX)
string(LENGTH "${hex}" length)
math(EXPR remainder "${length} % 8")
if(length EQUAL 0 OR NOT remainder EQUAL 0)
  message(FATAL_ERROR "${NSS_VK_SPV}: not a SPIR-V word stream (${length} hex digits)")
endif()
if(NOT hex MATCHES "^03022307")
  message(FATAL_ERROR "${NSS_VK_SPV}: missing the little-endian SPIR-V magic number")
endif()
math(EXPR words "${length} / 8")
# Little-endian bytes b0 b1 b2 b3 -> word 0xb3b2b1b0, eight words per line.
string(REGEX REPLACE "(..)(..)(..)(..)" "0x\\4\\3\\2\\1u," body "${hex}")
# (CMake regular expressions have no {n} repetition.)
set(word "0x[0-9a-f]+u,")
string(REGEX REPLACE "(${word}${word}${word}${word}${word}${word}${word}${word})" "\\1\n    " body "${body}")
file(WRITE "${NSS_VK_CPP}"
"// Generated from ${NSS_VK_ID}.spv by cmake/VulkanEmbed.cmake; do not edit.
#include <cstddef>
#include <cstdint>

namespace nss_vk::spirv {
extern const std::uint32_t ${NSS_VK_ID}[];
extern const std::size_t ${NSS_VK_ID}_words;
const std::uint32_t ${NSS_VK_ID}[] = {
    ${body}
};
const std::size_t ${NSS_VK_ID}_words = ${words};
}  // namespace nss_vk::spirv
")
