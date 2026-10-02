// SPDX-License-Identifier: GPL-2.0-only
#include "host/full_image.hpp"
#include "frontend/validate.hpp"
#include "frontend/full_image_args.hpp"

namespace {
void VS_CC create(const VSMap* in, VSMap* out, void*, VSCore* core, const VSAPI* api) {
    VSNode* node = nss_create_full_image(in, core, api, out, nss::Model::TWSC);
    if (node) api->mapConsumeNode(out, "clip", node, maAppend);
}
}
void register_twsc(VSPlugin* plugin, const VSPLUGINAPI* api) {
    const char* args = nss::frontend::kTwscSignature;
    api->registerFunction("TWSC", args, "clip:vnode;", nss::checked_create<create>, nullptr, plugin);
}
