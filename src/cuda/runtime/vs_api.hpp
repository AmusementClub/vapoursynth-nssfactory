// SPDX-License-Identifier: GPL-2.0-only
#pragma once

#include <VapourSynth4.h>

struct VSVULKANAPI;

namespace nss_cuda::vs_api {

bool gpu_frames(const VSAPI* vsapi);
bool gpu_frames(const VSPLUGINAPI* vspapi);
const VSVULKANAPI* vulkan(const VSAPI* vsapi);
bool gpu_node(const VSAPI* vsapi, VSNode* node);
VSNode* create_video_filter(const VSAPI* vsapi, const char* name, const VSVideoInfo* vi, VSFilterGetFrame get_frame,
                            VSFilterFree free, int mode, bool gpu_output, const VSFilterDependency* deps, int num_deps,
                            void* data, VSCore* core);

}  // namespace nss_cuda::vs_api
