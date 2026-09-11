#include "host/full_image.hpp"
#include "host/validate.hpp"

namespace {
void VS_CC create(const VSMap* in, VSMap* out, void*, VSCore* core, const VSAPI* api) {
    VSNode* node = nss_create_full_image(in, core, api, out, nss::Model::TWSC);
    if (node) api->mapConsumeNode(out, "clip", node, maAppend);
}
}
void register_twsc(VSPlugin* plugin, const VSPLUGINAPI* api) {
    const char* args =
        "clip:vnode;sigma:float[]:opt;block_size:int:opt;block_step:int:opt;group_size:int:opt;"
        "bm_range:int:opt;radius:int:opt;ps_num:int:opt;ps_range:int:opt;lambda2:float:opt;"
        "rclip:vnode:opt;iters:int:opt;delta:float:opt;estimate_sigma:int:opt;search_window:int:opt;"
        "admm_iter:int:opt;rho:float:opt;mu:float:opt;tol:float:opt;memory_limit_mb:int:opt;";
    api->registerFunction("TWSC", args, "clip:vnode;", nss::checked_create<create>, nullptr, plugin);
}
