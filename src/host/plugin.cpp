#include "nss/version.hpp"
#include "nss/backend.hpp"
#include "nss/cpu_lssc.hpp"

#include <VapourSynth4.h>
#include <VSHelper4.h>

void register_nlm(VSPlugin* plugin, const VSPLUGINAPI* vspapi);
void register_bm3d(VSPlugin* plugin, const VSPLUGINAPI* vspapi);
void register_vaggregate(VSPlugin* plugin, const VSPLUGINAPI* vspapi);
void register_wnnm(VSPlugin* plugin, const VSPLUGINAPI* vspapi);
void register_mcwnnm(VSPlugin* plugin, const VSPLUGINAPI* vspapi);
void register_twsc(VSPlugin* plugin, const VSPLUGINAPI* vspapi);
void register_nlh(VSPlugin* plugin, const VSPLUGINAPI* vspapi);
void register_ncsr(VSPlugin* plugin, const VSPLUGINAPI* vspapi);
void register_lssc(VSPlugin* plugin, const VSPLUGINAPI* vspapi);

static void VS_CC versionCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi) {
    (void)in;
    (void)userData;
    (void)core;
    vsapi->mapSetData(out, "version", nss::version_string(), -1, dtUtf8, maReplace);
}

static void VS_CC backendCreate(const VSMap*, VSMap* out, void*, VSCore*, const VSAPI* vsapi) {
    const auto caps = nss::backend_caps();
    vsapi->mapSetInt(out, "compiled_targets", caps.compiled_targets, maReplace);
    vsapi->mapSetInt(out, "runtime_targets", caps.runtime_targets, maReplace);
    vsapi->mapSetInt(out, "selected_target", caps.selected_target, maReplace);
    vsapi->mapSetInt(out, "float_lanes", caps.float_lanes, maReplace);
    vsapi->mapSetInt(out, "executable", caps.executable, maReplace);
    vsapi->mapSetInt(out, "portable_test", caps.portable_test, maReplace);
    vsapi->mapSetData(out, "target_name", caps.target_name, -1, dtUtf8, maReplace);
    vsapi->mapSetData(out, "build_mode", caps.build_mode, -1, dtUtf8, maReplace);
    vsapi->mapSetInt(out, "lssc_sme_compiled", nss::lssc_sme_compiled(), maReplace);
    vsapi->mapSetInt(out, "lssc_sme_available", nss::lssc_sme_available(), maReplace);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->configPlugin("com.nssfactory.nss", "nss", "NSS denoising factory (CPU)", VS_MAKE_VERSION(1, 0),
                         VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("Version", "", "version:data;", versionCreate, nullptr, plugin);
    vspapi->registerFunction("Backend", "",
        "compiled_targets:int;runtime_targets:int;selected_target:int;float_lanes:int;"
        "executable:int;portable_test:int;target_name:data;build_mode:data;"
        "lssc_sme_compiled:int;lssc_sme_available:int;", backendCreate, nullptr, plugin);
    register_nlm(plugin, vspapi);
    register_bm3d(plugin, vspapi);
    register_vaggregate(plugin, vspapi);
    register_wnnm(plugin, vspapi);
    register_mcwnnm(plugin, vspapi);
    register_twsc(plugin, vspapi);
    register_nlh(plugin, vspapi);
    register_ncsr(plugin, vspapi);
    register_lssc(plugin, vspapi);
}
