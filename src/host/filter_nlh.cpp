// SPDX-License-Identifier: GPL-2.0-only
#include "host/full_image.hpp"
#include "frontend/validate.hpp"
#include "frontend/full_image_args.hpp"

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
    const char* args = nss::frontend::kNlhSignature;
    api->registerFunction("NLH", args, "clip:vnode;", nss::checked_create<create>, nullptr, plugin);
}
