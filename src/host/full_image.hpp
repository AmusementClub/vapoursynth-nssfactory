#pragma once
#include <VapourSynth4.h>
#include "nss/contracts.hpp"

VSNode* nss_create_full_image(const VSMap* in, VSCore* core, const VSAPI* api, VSMap* error, nss::Model model);
