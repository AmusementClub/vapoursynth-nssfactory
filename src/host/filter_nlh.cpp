#include "host/full_image.hpp"
#include "host/validate.hpp"

VSNode* nss_create_nlh(const VSMap* in, VSCore* core, const VSAPI* api, VSMap* error) {
    return nss_create_full_image(in, core, api, error, nss::Model::NLH);
}
namespace {
void VS_CC create(const VSMap* in, VSMap* out, void*, VSCore* core, const VSAPI* api) {
    VSNode* node = nss_create_nlh(in, core, api, out);
    if (node) api->mapConsumeNode(out, "clip", node, maAppend);
}
}
void register_nlh(VSPlugin* plugin, const VSPLUGINAPI* api) {
    const char* args =
        "clip:vnode;sigma:float[]:opt;block_size:int[]:opt;block_step:int[]:opt;group_size:int[]:opt;"
        "bm_range:int:opt;radius:int:opt;ps_num:int:opt;ps_range:int:opt;q:int[]:opt;rclip:vnode:opt;"
        "noise_model:data:opt;search_window:int[]:opt;basic_iters:int:opt;lambda_basic:float:opt;"
        "hard_strength:float:opt;wiener_iters:int:opt;wiener_sigma_scale:float:opt;memory_limit_mb:int:opt;";
    api->registerFunction("NLH", args, "clip:vnode;", nss::checked_create<create>, nullptr, plugin);
}
