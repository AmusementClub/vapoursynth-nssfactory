#pragma once

#include <VapourSynth4.h>

// Returns a node the caller owns (refcount 1), or nullptr after mapSetError on err.
// nss_create_vaggregate takes ownership of fat and src.
VSNode* nss_create_bm3d(const VSMap* in, VSCore* core, const VSAPI* vsapi, VSMap* err);
VSNode* nss_create_wnnm(const VSMap* in, VSCore* core, const VSAPI* vsapi, VSMap* err);
VSNode* nss_create_vaggregate(VSNode* fat, VSNode* src, int radius, const int* planes, VSCore* core,
                              const VSAPI* vsapi, VSMap* err);
