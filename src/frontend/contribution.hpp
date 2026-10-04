// SPDX-License-Identifier: GPL-2.0-only
#pragma once
#include <VapourSynth4.h>
#include "nss/contracts.hpp"
#include <stdexcept>
#include <string>

namespace nss {
inline void contribution_set(const VSAPI* api, VSMap* props, const char* key, int value) {
    if (api->mapSetInt(props, key, value, maReplace)) throw std::bad_alloc();
}
inline void stamp_contribution(VSFrame* frame, int radius, int center, Model model, const VSAPI* api) {
    auto* props = api->getFramePropertiesRW(frame);
    const int profile = model == Model::BM3D ? NoiseProfile::bm3d : 0;
    contribution_set(api, props, "_NSSModel", static_cast<int>(model));
    contribution_set(api, props, "_NSSModelVersion", model_semantic_version(model));
    contribution_set(api, props, "_NSSNoiseProfile", profile);
    if (radius > 0) {
        contribution_set(api, props, "_NSSFatVersion", kContributionVersion);
        contribution_set(api, props, "_NSSFatRadius", radius);
        contribution_set(api, props, "_NSSFatCenter", center);
        contribution_set(api, props, "_NSSFatLayout", kContributionNumDenSlices);
    } else {
        for (const auto* key : {"_NSSFatVersion", "_NSSFatRadius", "_NSSFatCenter", "_NSSFatLayout"})
            api->mapDeleteKey(props, key);
    }
}
// ns is the plugin namespace used in error text ("nss", "nss_cuda").
inline int contribution_property(const VSMap* props, const char* key, const VSAPI* api, const char* ns = "nss") {
    int error = 0;
    const auto value = api->mapGetInt(props, key, 0, &error);
    if (error || value < 0 || value > 2147483647)
        throw std::invalid_argument(std::string(ns) + ".VAggregate: missing or invalid contribution identity");
    return static_cast<int>(value);
}
inline ContributionLayout validate_contribution(const VSFrame* frame, int radius, int center, bool allow_legacy,
                                                const VSAPI* api, const char* ns = "nss") {
    const auto* props = api->getFramePropertiesRO(frame);
    if (api->mapNumElements(props, "_NSSFatVersion") < 0 && allow_legacy) {
        // Only wholly untagged inputs may enter the explicit legacy adapter.
        for (const auto* key : {"_NSSFatRadius", "_NSSFatCenter", "_NSSFatLayout", "_NSSModel", "_NSSModelVersion", "_NSSNoiseProfile"})
            if (api->mapNumElements(props, key) >= 0)
                throw std::invalid_argument(std::string(ns) + ".VAggregate: partial contribution identity");
        return {0, radius, center, kContributionNumDenSlices, 0, 0};
    }
    ContributionLayout result{
        contribution_property(props, "_NSSFatVersion", api, ns),
        contribution_property(props, "_NSSFatRadius", api, ns),
        contribution_property(props, "_NSSFatCenter", api, ns),
        contribution_property(props, "_NSSFatLayout", api, ns),
        contribution_property(props, "_NSSModel", api, ns),
        contribution_property(props, "_NSSNoiseProfile", api, ns)};
    if (result.version != kContributionVersion || result.radius != radius || result.center != center ||
        result.slice_layout != kContributionNumDenSlices || result.model < 1 || result.model > 6 ||
        result.profile != (result.model == static_cast<int>(Model::BM3D) ? NoiseProfile::bm3d : 0) ||
        contribution_property(props, "_NSSModelVersion", api, ns) != model_semantic_version(static_cast<Model>(result.model)))
        throw std::invalid_argument(std::string(ns) + ".VAggregate: incompatible contribution identity");
    return result;
}
} // namespace nss
