// SPDX-License-Identifier: GPL-2.0-only
// The VSAPI members of API 4.3 (VapourSynth R80+ GPU frames, D29), behind
// plain functions. libnss_vk is one binary for every core: it declares API
// 4.0 (R78 and vapoursynth-classic refuse a plugin that declares 4.3), and
// only runtime/vs_api43.cpp is compiled with VS_USE_API_43, where these
// functions read the 4.3 members once getAPIVersion() reports 4.3. Every
// other file sees the 4.0 VSAPI. Built without VSVulkan4.h
// (NSS_VK_GPU_FRAMES=0) they report CPU frames only.
#pragma once

#include <VapourSynth4.h>

struct VSVULKANAPI;

namespace nss_vk::vs_api {

// The core offers API 4.3 (GPU frames) and this build supports it.
bool gpu_frames(const VSAPI* vsapi);
bool gpu_frames(const VSPLUGINAPI* vspapi);
// VSAPI::getVulkanAPI; null without API 4.3.
const VSVULKANAPI* vulkan(const VSAPI* vsapi);
// The node's or frame's residency is the GPU (false without API 4.3).
bool gpu_node(const VSAPI* vsapi, VSNode* node);
bool gpu_frame(const VSAPI* vsapi, const VSFrame* frame);
// createVideoFilter2, or createVideoFilterEx2 with ffGPUOutput when
// `gpu_output` (API 4.3 only).
VSNode* create_video_filter(const VSAPI* vsapi, const char* name, const VSVideoInfo* vi, VSFilterGetFrame get_frame,
                            VSFilterFree free, int mode, bool gpu_output, const VSFilterDependency* deps, int num_deps,
                            void* data, VSCore* core);

}  // namespace nss_vk::vs_api
