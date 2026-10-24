// SPDX-License-Identifier: GPL-2.0-only
// libnss_vk entry point (namespace nss_vk; docs/vulkan-plan.md). Filters join
// phase by phase from V4 on; each will reuse its frontend parse_<filter> with
// the "nss_vk" prefix and append device_id and num_streams (D21).
#include "nss/version.hpp"
#include "vulkan/runtime/device.hpp"
#include "vulkan/runtime/instance.hpp"
#include "vulkan/runtime/probe.hpp"
#include "vulkan/runtime/shaders.hpp"

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include <exception>
#include <string>

namespace nss_vk {
void register_nlm(VSPlugin* plugin, const VSPLUGINAPI* vspapi);
}  // namespace nss_vk

namespace {

void VS_CC versionCreate(const VSMap*, VSMap* out, void*, VSCore*, const VSAPI* vsapi) {
    vsapi->mapSetData(out, "version", nss::version_string(), -1, dtUtf8, maReplace);
}

void set_text(const VSAPI* vsapi, VSMap* out, const char* key, const std::string& value) {
    vsapi->mapSetData(out, key, value.c_str(), static_cast<int>(value.size()), dtUtf8, maReplace);
}

void VS_CC backendCreate(const VSMap* in, VSMap* out, void*, VSCore*, const VSAPI* vsapi) {
    const nss_vk::Instance& vk = nss_vk::instance();
    const int count = static_cast<int>(vk.devices.size());
    set_text(vsapi, out, "loader_version", nss_vk::version_text(vk.loader_version));
    vsapi->mapSetInt(out, "device_count", count, maReplace);
    set_text(vsapi, out, "error", vk.error);
    set_text(vsapi, out, "shader_tools", nss_vk::shader_tools());

    int err = 0;
    const int requested = vsapi->mapGetIntSaturated(in, "device_id", 0, &err);
    const bool explicit_device = !err;
    if (count == 0) {
        if (explicit_device) {
            vsapi->mapSetError(out, ("nss_vk.Backend: no Vulkan device available (" + vk.error + ")").c_str());
        }
        return;
    }
    const int index = explicit_device ? requested : 0;
    if (index < 0 || index >= count) {
        vsapi->mapSetError(out, ("nss_vk.Backend: device_id " + std::to_string(index) + " out of range (" +
                                 std::to_string(count) + " Vulkan devices)").c_str());
        return;
    }
    try {
        VkPhysicalDevice device = vk.devices[static_cast<std::size_t>(index)];
        const nss_vk::DeviceCaps caps = nss_vk::query_caps(device);
        const nss_vk::Classification support = nss_vk::classify(caps);
        const nss_vk::ProbeResult probe = nss_vk::probe(device, caps);
        vsapi->mapSetInt(out, "device", index, maReplace);
        set_text(vsapi, out, "name", caps.name);
        set_text(vsapi, out, "device_type", nss_vk::device_type_name(caps.type));
        vsapi->mapSetInt(out, "vendor_id", caps.vendor_id, maReplace);
        set_text(vsapi, out, "api_version", nss_vk::version_text(caps.api_version));
        set_text(vsapi, out, "driver_name", caps.driver_name);
        set_text(vsapi, out, "driver_info", caps.driver_info);
        set_text(vsapi, out, "support", nss_vk::support_name(support.support));
        set_text(vsapi, out, "missing", nss_vk::join(support.missing));
        vsapi->mapSetInt(out, "native_lanes", support.native_lanes, maReplace);
        vsapi->mapSetInt(out, "subgroup_size", caps.subgroup_size, maReplace);
        vsapi->mapSetInt(out, "min_subgroup_size", caps.min_subgroup_size, maReplace);
        vsapi->mapSetInt(out, "max_subgroup_size", caps.max_subgroup_size, maReplace);
        vsapi->mapSetInt(out, "shared_memory_kb", caps.shared_memory / 1024, maReplace);
        vsapi->mapSetInt(out, "total_memory_mb", static_cast<int64_t>(caps.device_local_bytes >> 20), maReplace);
        vsapi->mapSetInt(out, "host_visible_device_mb", static_cast<int64_t>(caps.host_visible_device_bytes >> 20),
                         maReplace);
        vsapi->mapSetInt(out, "transfer_queues", caps.transfer_queues, maReplace);
        set_text(vsapi, out, "optional", nss_vk::join(nss_vk::optional_features(caps)));
        vsapi->mapSetInt(out, "probe_ok", probe.ok, maReplace);
        vsapi->mapSetFloat(out, "probe_ms", probe.milliseconds, maReplace);
        vsapi->mapSetInt(out, "probe_subgroup_size", probe.subgroup_size, maReplace);
        set_text(vsapi, out, "probe_error", probe.error);
    } catch (const std::exception& error) {
        vsapi->mapSetError(out, (std::string("nss_vk.Backend: ") + error.what()).c_str());
    }
}

}  // namespace

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->configPlugin("com.nssfactory.nss_vk", "nss_vk", "NSS denoising factory (Vulkan)", VS_MAKE_VERSION(1, 0),
                         VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("Version", "", "version:data;", versionCreate, nullptr, plugin);
    vspapi->registerFunction("Backend", "device_id:int:opt;",
        "loader_version:data;device_count:int;error:data;shader_tools:data;"
        "device:int:opt;name:data:opt;device_type:data:opt;vendor_id:int:opt;api_version:data:opt;"
        "driver_name:data:opt;driver_info:data:opt;support:data:opt;missing:data:opt;native_lanes:int:opt;"
        "subgroup_size:int:opt;min_subgroup_size:int:opt;max_subgroup_size:int:opt;shared_memory_kb:int:opt;"
        "total_memory_mb:int:opt;host_visible_device_mb:int:opt;transfer_queues:int:opt;optional:data:opt;"
        "probe_ok:int:opt;probe_ms:float:opt;probe_subgroup_size:int:opt;probe_error:data:opt;",
        backendCreate, nullptr, plugin);
    nss_vk::register_nlm(plugin, vspapi);
}
