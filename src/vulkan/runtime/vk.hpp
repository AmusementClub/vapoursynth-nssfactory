// SPDX-License-Identifier: GPL-2.0-only
// The single include point for Vulkan in src/vulkan: volk declares every entry
// point as a function pointer (VK_NO_PROTOTYPES), so the plugin loads on
// machines without a Vulkan loader and reports why at use (D25).
#pragma once

#include <volk.h>
