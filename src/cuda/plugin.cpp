// SPDX-License-Identifier: GPL-2.0-only
// libnss_cuda entry point (namespace nss_cuda). Filters are registered phase
// by phase per docs/cuda-plan.md; each one reuses its frontend parse_<filter>
// with the "nss_cuda" prefix and the backend argument tail (D14).
#include "cuda/runtime/device.hpp"
#include "nss/version.hpp"

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include <exception>
#include <string>

namespace {

void VS_CC versionCreate(const VSMap*, VSMap* out, void*, VSCore*, const VSAPI* vsapi) {
    vsapi->mapSetData(out, "version", nss::version_string(), -1, dtUtf8, maReplace);
}

void VS_CC backendCreate(const VSMap* in, VSMap* out, void*, VSCore*, const VSAPI* vsapi) {
    const nss_cuda::RuntimeInfo runtime = nss_cuda::runtime_info();
    vsapi->mapSetInt(out, "device_count", runtime.device_count, maReplace);
    vsapi->mapSetInt(out, "driver_version", runtime.driver_version, maReplace);
    vsapi->mapSetInt(out, "runtime_version", runtime.runtime_version, maReplace);
    vsapi->mapSetData(out, "compiled_architectures", nss_cuda::compiled_architectures(), -1, dtUtf8, maReplace);
    vsapi->mapSetData(out, "error", runtime.error.c_str(), -1, dtUtf8, maReplace);
    vsapi->mapSetInt(out, "ptx_jit_available", runtime.ptx_jit_available(), maReplace);

    int err = 0;
    const int requested = vsapi->mapGetIntSaturated(in, "device_id", 0, &err);
    const bool explicit_device = !err;
    if (runtime.device_count == 0) {
        if (explicit_device) {
            vsapi->mapSetError(out, ("nss_cuda.Backend: no CUDA device available (" + runtime.error + ")").c_str());
        }
        return;
    }
    try {
        const nss_cuda::DeviceInfo device = nss_cuda::device_info(explicit_device ? requested : 0, runtime);
        const nss_cuda::ProbeResult probe = nss_cuda::probe(device.index);
        const std::string cc = std::to_string(device.cc_major) + "." + std::to_string(device.cc_minor);
        vsapi->mapSetInt(out, "device", device.index, maReplace);
        vsapi->mapSetData(out, "name", device.name.c_str(), -1, dtUtf8, maReplace);
        vsapi->mapSetData(out, "compute_capability", cc.c_str(), -1, dtUtf8, maReplace);
        vsapi->mapSetData(out, "support", nss_cuda::support_name(device.support), -1, dtUtf8, maReplace);
        vsapi->mapSetInt(out, "total_memory_mb", static_cast<int64_t>(device.total_memory >> 20), maReplace);
        vsapi->mapSetInt(out, "multiprocessors", device.multiprocessors, maReplace);
        vsapi->mapSetInt(out, "probe_ok", probe.ok, maReplace);
        vsapi->mapSetFloat(out, "probe_ms", probe.milliseconds, maReplace);
        vsapi->mapSetData(out, "probe_error", probe.error.c_str(), -1, dtUtf8, maReplace);
    } catch (const std::exception& error) {
        vsapi->mapSetError(out, (std::string("nss_cuda.Backend: ") + error.what()).c_str());
    }
}

}  // namespace

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->configPlugin("com.nssfactory.nss_cuda", "nss_cuda", "NSS denoising factory (CUDA)", VS_MAKE_VERSION(1, 0),
                         VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("Version", "", "version:data;", versionCreate, nullptr, plugin);
    vspapi->registerFunction("Backend", "device_id:int:opt;",
        "device_count:int;driver_version:int;runtime_version:int;compiled_architectures:data;error:data;"
        "ptx_jit_available:int;"
        "device:int:opt;name:data:opt;compute_capability:data:opt;support:data:opt;total_memory_mb:int:opt;"
        "multiprocessors:int:opt;probe_ok:int:opt;probe_ms:float:opt;probe_error:data:opt;",
        backendCreate, nullptr, plugin);
}
