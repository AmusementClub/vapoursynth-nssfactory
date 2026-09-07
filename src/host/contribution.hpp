#pragma once
#include <VapourSynth4.h>
#include "nss/contracts.hpp"
#include <stdexcept>

namespace nss {
inline void contribution_set(const VSAPI* api, VSMap* props, const char* key, int value) {
    if (api->mapSetInt(props, key, value, maReplace)) throw std::bad_alloc();
}
inline void stamp_contribution(VSFrame* frame, int radius, int center, Model model, const VSAPI* api) {
    auto* props = api->getFramePropertiesRW(frame);
    const int profile = model == Model::BM3D ? NoiseProfile::bm3d : 0;
    contribution_set(api, props, "_NSSModel", static_cast<int>(model));
    contribution_set(api, props, "_NSSModelVersion", kSemanticVersion);
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
inline int contribution_property(const VSMap* props, const char* key, const VSAPI* api) {
    int error = 0;
    const auto value = api->mapGetInt(props, key, 0, &error);
    if (error || value < 0 || value > 2147483647)
        throw std::invalid_argument("nss.VAggregate: missing or invalid contribution identity");
    return static_cast<int>(value);
}
inline ContributionLayout validate_contribution(const VSFrame* frame, int radius, int center, bool allow_legacy,
                                                const VSAPI* api) {
    const auto* props = api->getFramePropertiesRO(frame);
    if (api->mapNumElements(props, "_NSSFatVersion") < 0 && allow_legacy) {
        // Only wholly untagged inputs may enter the explicit legacy adapter.
        for (const auto* key : {"_NSSFatRadius", "_NSSFatCenter", "_NSSFatLayout", "_NSSModel", "_NSSNoiseProfile"})
            if (api->mapNumElements(props, key) >= 0)
                throw std::invalid_argument("nss.VAggregate: partial contribution identity");
        return {0, radius, center, kContributionNumDenSlices, 0, 0};
    }
    ContributionLayout result{
        contribution_property(props, "_NSSFatVersion", api),
        contribution_property(props, "_NSSFatRadius", api),
        contribution_property(props, "_NSSFatCenter", api),
        contribution_property(props, "_NSSFatLayout", api),
        contribution_property(props, "_NSSModel", api),
        contribution_property(props, "_NSSNoiseProfile", api)};
    if (result.version != kContributionVersion || result.radius != radius || result.center != center ||
        result.slice_layout != kContributionNumDenSlices || result.model < 1 || result.model > 6 ||
        result.profile != (result.model == static_cast<int>(Model::BM3D) ? NoiseProfile::bm3d : 0) ||
        contribution_property(props, "_NSSModelVersion", api) != kSemanticVersion)
        throw std::invalid_argument("nss.VAggregate: incompatible contribution identity");
    return result;
}
} // namespace nss
