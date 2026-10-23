// SPDX-License-Identifier: GPL-2.0-only
// The single include point for Vulkan in src/vulkan: volk declares every entry
// point as a function pointer (VK_NO_PROTOTYPES), so the plugin loads on
// machines without a Vulkan loader and reports why at use (D25). VMA takes its
// entry points from volk at allocator creation (runtime/vma.cpp).
#pragma once

#include <volk.h>

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>
