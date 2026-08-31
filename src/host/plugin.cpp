#include "nss/version.hpp"

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

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->configPlugin("com.nssfactory.nss", "nss", "NSS denoising factory (CPU)", VS_MAKE_VERSION(1, 0),
                         VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("Version", "", "version:data;", versionCreate, nullptr, plugin);
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
