// SPDX-License-Identifier: GPL-2.0-only
// Compiled with VS_USE_API_43 (src/vulkan/CMakeLists.txt) when VSVulkan4.h
// is there: the only file that sees the API 4.3 members of VSAPI. See
// vs_api.hpp. The VSAPI the core hands a plugin is the same table whatever
// API the plugin declared (R81 vscore.cpp getVapourSynthAPI), and the
// members of a minor version are appended after those of the one before, so
// reading them is safe once getAPIVersion() reports 4.3.
#include "vulkan/runtime/vs_api.hpp"

namespace nss_vk::vs_api {

#if NSS_VK_GPU_FRAMES
namespace {
constexpr int kApi43 = VS_MAKE_VERSION(4, 3);
static_assert(VAPOURSYNTH_API_MINOR >= 3, "runtime/vs_api43.cpp must be compiled with VS_USE_API_43");
}  // namespace

bool gpu_frames(const VSAPI* vsapi) { return vsapi && vsapi->getAPIVersion() >= kApi43; }
bool gpu_frames(const VSPLUGINAPI* vspapi) { return vspapi && vspapi->getAPIVersion() >= kApi43; }

const VSVULKANAPI* vulkan(const VSAPI* vsapi) { return gpu_frames(vsapi) ? vsapi->getVulkanAPI() : nullptr; }

bool gpu_node(const VSAPI* vsapi, VSNode* node) {
    return node && gpu_frames(vsapi) && vsapi->getNodeResidency(node) == nrGPU;
}

bool gpu_frame(const VSAPI* vsapi, const VSFrame* frame) {
    return frame && gpu_frames(vsapi) && vsapi->getFrameResidency(frame) == nrGPU;
}

VSNode* create_video_filter(const VSAPI* vsapi, const char* name, const VSVideoInfo* vi, VSFilterGetFrame get_frame,
                            VSFilterFree free, int mode, bool gpu_output, const VSFilterDependency* deps, int num_deps,
                            void* data, VSCore* core) {
    if (gpu_output && gpu_frames(vsapi)) {
        return vsapi->createVideoFilterEx2(name, vi, get_frame, free, mode, ffGPUOutput, deps, num_deps, data, core);
    }
    return vsapi->createVideoFilter2(name, vi, get_frame, free, mode, deps, num_deps, data, core);
}
#else
bool gpu_frames(const VSAPI*) { return false; }
bool gpu_frames(const VSPLUGINAPI*) { return false; }
const VSVULKANAPI* vulkan(const VSAPI*) { return nullptr; }
bool gpu_node(const VSAPI*, VSNode*) { return false; }
bool gpu_frame(const VSAPI*, const VSFrame*) { return false; }

VSNode* create_video_filter(const VSAPI* vsapi, const char* name, const VSVideoInfo* vi, VSFilterGetFrame get_frame,
                            VSFilterFree free, int mode, bool, const VSFilterDependency* deps, int num_deps, void* data,
                            VSCore* core) {
    return vsapi->createVideoFilter2(name, vi, get_frame, free, mode, deps, num_deps, data, core);
}
#endif

}  // namespace nss_vk::vs_api
