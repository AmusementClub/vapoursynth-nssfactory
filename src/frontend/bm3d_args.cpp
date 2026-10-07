// SPDX-License-Identifier: GPL-2.0-only
#include "frontend/bm3d_args.hpp"
#include "frontend/args.hpp"
#include "frontend/validate.hpp"
#include "nss/contracts.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace nss::frontend {

Bm3dParams parse_bm3d(const VSAPI* vsapi, const VSMap* in, const VSVideoInfo& vi, const VSVideoInfo* ref_vi,
                      const char* ns) {
    if (!is_const_32f(vi)) fail(ns, "BM3D", "constant Gray/YUV/RGB 32-bit float required");
    if (ref_vi && !same_video(vi, *ref_vi)) fail(ns, "BM3D", "ref must match clip");
    Bm3dParams p;
    const int np = vi.format.numPlanes;
    map_float_array(vsapi, in, "sigma", p.sigma, np, kBmDefaultSigma);
    float sigma_given[3]{p.sigma[0], p.sigma[1], p.sigma[2]};
    for (int i = 0; i < np; ++i) {
        if (!finite_bits(p.sigma[i]) || p.sigma[i] < 0.f) fail(ns, "BM3D", "sigma must be finite and non-negative");
        if (p.sigma[i] != 0.f) p.sigma[i] = NoiseProfile::bm3d_effective(p.sigma[i]);
    }
    map_inherit_int(vsapi, in, "block_size", p.block_size, np, kBmBlock);
    map_inherit_int(vsapi, in, "group_size", p.group_size, np, kBmGroup);
    if (vsapi->mapNumElements(in, "block_step") <= 0) {
        for (int i = 0; i < np; ++i) p.block_step[i] = std::min(kBmDefaultStep, p.block_size[i]);
    } else {
        map_inherit_int(vsapi, in, "block_step", p.block_step, np, kBmDefaultStep);
    }
    map_int_array(vsapi, in, "bm_range", p.bm_range, np, kBmDefaultRange);
    if (vsapi->mapNumElements(in, "ps_num") <= 0) {
        for (int i = 0; i < np; ++i) p.ps_num[i] = std::min(kBmDefaultPsNum, p.group_size[i]);
    } else {
        map_inherit_int(vsapi, in, "ps_num", p.ps_num, np, kBmDefaultPsNum);
    }
    map_int_array(vsapi, in, "ps_range", p.ps_range, np, kBmDefaultPsRange);
    p.radius = map_int(vsapi, in, "radius", 0);
    if (p.radius < 0 || p.radius > kBmMaxRadius) fail(ns, "BM3D", "radius must be in [0, 16]");
    const int chroma = map_int(vsapi, in, "chroma", 0);
    if (chroma != 0 && chroma != 1) fail(ns, "BM3D", "chroma must be 0 or 1");
    p.chroma = chroma != 0;
    if (p.chroma) {
        if (vi.format.colorFamily != cfYUV || vi.format.subSamplingW != 0 || vi.format.subSamplingH != 0) {
            fail(ns, "BM3D", "chroma requires a YUV 4:4:4 clip");
        }
        // One set of groups, found on plane 0, serves the three planes.
        for (int i = 1; i < np; ++i) {
            p.block_size[i] = p.block_size[0];
            p.group_size[i] = p.group_size[0];
            p.block_step[i] = p.block_step[0];
            p.bm_range[i] = p.bm_range[0];
            p.ps_num[i] = p.ps_num[0];
            p.ps_range[i] = p.ps_range[0];
        }
    }
    const int final = map_int(vsapi, in, "final", 0);
    if (final != 0 && final != 1) fail(ns, "BM3D", "final must be 0 or 1");
    p.final = final != 0;
    const bool sigma_basic = vsapi->mapNumElements(in, "sigma_basic") > 0;
    const bool block_basic = vsapi->mapNumElements(in, "block_size_basic") > 0;
    const bool group_basic = vsapi->mapNumElements(in, "group_size_basic") > 0;
    if (!p.final && (sigma_basic || block_basic || group_basic)) {
        fail(ns, "BM3D", "sigma_basic, block_size_basic and group_size_basic require final=1");
    }
    if (p.final && ref_vi) fail(ns, "BM3D", "use only one of final and ref");
    if (p.final) {
        // The basic stage: its own sigma, block and group where given, the
        // call's otherwise; block_step and ps_num as given, or adapted to it.
        if (sigma_basic) map_float_array(vsapi, in, "sigma_basic", sigma_given, np, kBmDefaultSigma);
        const bool step_given = vsapi->mapNumElements(in, "block_step") > 0;
        const bool ps_given = vsapi->mapNumElements(in, "ps_num") > 0;
        for (int i = 0; i < np; ++i) {
            p.block_size_basic[i] = p.block_size[i];
            p.group_size_basic[i] = p.group_size[i];
        }
        if (block_basic) map_inherit_int(vsapi, in, "block_size_basic", p.block_size_basic, np, kBmBlock);
        if (group_basic) map_inherit_int(vsapi, in, "group_size_basic", p.group_size_basic, np, kBmGroup);
        for (int i = 0; i < np; ++i) {
            const int plane = p.chroma ? 0 : i;
            p.block_size_basic[i] = p.block_size_basic[plane];
            p.group_size_basic[i] = p.group_size_basic[plane];
            if (!finite_bits(sigma_given[i]) || sigma_given[i] < 0.f) {
                fail(ns, "BM3D", "sigma_basic must be finite and non-negative");
            }
            p.sigma_basic[i] = sigma_given[i] != 0.f ? NoiseProfile::bm3d_effective(sigma_given[i]) : 0.f;
            if (!bm_allowed_block(p.block_size_basic[i])) {
                fail(ns, "BM3D", "block_size_basic must be one of 1, 2, 4, 8, 12, 16, 32");
            }
            if (!bm_allowed_group(p.group_size_basic[i])) {
                fail(ns, "BM3D", "group_size_basic must be one of 1, 2, 4, 8, 16, 32, 64");
            }
            if (p.block_size_basic[i] > plane_width(vi, i) || p.block_size_basic[i] > plane_height(vi, i)) {
                fail(ns, "BM3D", "block_size_basic must not exceed plane dimensions");
            }
            p.block_step_basic[i] = step_given ? p.block_step[i] : std::min(kBmDefaultStep, p.block_size_basic[i]);
            p.ps_num_basic[i] = ps_given ? p.ps_num[i] : std::min(kBmDefaultPsNum, p.group_size_basic[i]);
            if (p.block_step_basic[i] > p.block_size_basic[i]) {
                fail(ns, "BM3D", "block_step must be in [1, block_size_basic]");
            }
            if (p.ps_num_basic[i] > p.group_size_basic[i]) fail(ns, "BM3D", "ps_num must be in [1, group_size_basic]");
        }
    }
    for (int i = 0; i < np; ++i) {
        if (!bm_allowed_block(p.block_size[i])) fail(ns, "BM3D", "block_size must be one of 1, 2, 4, 8, 12, 16, 32");
        if (!bm_allowed_group(p.group_size[i])) fail(ns, "BM3D", "group_size must be one of 1, 2, 4, 8, 16, 32, 64");
        if (p.block_size[i] > plane_width(vi, i) || p.block_size[i] > plane_height(vi, i))
            fail(ns, "BM3D", "block_size must not exceed plane dimensions");
        if (p.block_step[i] < 1 || p.block_step[i] > p.block_size[i])
            fail(ns, "BM3D", "block_step must be in [1, block_size]");
        if (p.ps_num[i] < 1 || p.ps_num[i] > p.group_size[i]) fail(ns, "BM3D", "ps_num must be in [1, group_size]");
        if (p.bm_range[i] < 1 || p.bm_range[i] > kBmMaxRange) fail(ns, "BM3D", "bm_range must be in [1, 64]");
        if (p.ps_range[i] < 0 || p.ps_range[i] > kBmMaxRange) fail(ns, "BM3D", "ps_range must be in [0, 64]");
    }
    return p;
}

namespace {

// A copy of `in` without the two-stage arguments.
VSMap* one_stage_args(const VSAPI* vsapi, const VSMap* in) {
    VSMap* args = vsapi->createMap();
    vsapi->copyMap(in, args);
    for (const char* key : {"final", "sigma_basic", "block_size_basic", "group_size_basic"}) {
        vsapi->mapDeleteKey(args, key);
    }
    return args;
}

}  // namespace

void bm3d_two_nodes(const VSAPI* vsapi, const VSMap* in, VSMap* out, VSCore* core, const char* ns) {
    VSPlugin* plugin = vsapi->getPluginByNamespace(ns, core);
    if (!plugin) fail(ns, "BM3D", "final could not find the plugin");
    const auto invoke = [&](VSMap* args, const char* stage) {
        VSMap* result = vsapi->invoke(plugin, "BM3D", args);
        vsapi->freeMap(args);
        if (const char* error = vsapi->mapGetError(result)) {
            const std::string message = std::string(ns) + ".BM3D: the " + stage + " stage failed: " + error;
            vsapi->freeMap(result);
            throw std::runtime_error(message);
        }
        VSNode* node = vsapi->mapGetNode(result, "clip", 0, nullptr);
        vsapi->freeMap(result);
        return node;
    };
    VSMap* basic = one_stage_args(vsapi, in);
    // The basic stage takes its values in place of the call's and returns finished frames.
    for (const char* name : {"sigma", "block_size", "group_size"}) {
        const std::string from = std::string(name) + "_basic";
        const int n = vsapi->mapNumElements(in, from.c_str());
        if (n <= 0) continue;
        vsapi->mapDeleteKey(basic, name);
        for (int i = 0; i < n; ++i) {
            if (name == std::string("sigma")) {
                vsapi->mapSetFloat(basic, name, vsapi->mapGetFloat(in, from.c_str(), i, nullptr), maAppend);
            } else {
                vsapi->mapSetInt(basic, name, vsapi->mapGetInt(in, from.c_str(), i, nullptr), maAppend);
            }
        }
    }
    if (map_int(vsapi, in, "radius", 0) > 0) vsapi->mapSetData(basic, "temporal_mode", "rolling", -1, dtUtf8, maReplace);
    VSNode* estimate = invoke(basic, "basic");
    VSMap* second = one_stage_args(vsapi, in);
    vsapi->mapConsumeNode(second, "ref", estimate, maReplace);
    vsapi->mapConsumeNode(out, "clip", invoke(second, "final"), maReplace);
}

}  // namespace nss::frontend
