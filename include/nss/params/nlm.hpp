// SPDX-License-Identifier: GPL-2.0-only
#pragma once
// Resolved NLM arguments shared by every backend.
#include "nss/params.hpp"

namespace nss {

struct NlmParams {
    int d = kNlmDefaultD;
    int a = kNlmDefaultA;
    int s = kNlmDefaultS;
    float h = kNlmDefaultH;
    float wref = kNlmDefaultWref;
    ChannelMode channels = ChannelMode::Y;
};

}  // namespace nss
