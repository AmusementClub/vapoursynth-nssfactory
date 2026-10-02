// SPDX-License-Identifier: GPL-2.0-only
#include "frontend/full_image_args.hpp"
#include "frontend/args.hpp"
#include "frontend/validate.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace nss::frontend {
namespace {

[[noreturn]] void fail_plain(const char* ns, const std::string& message) {
    throw std::invalid_argument(std::string(ns) + ": " + message);
}
bool present(const VSAPI* api, const VSMap* in, const char* key) { return api->mapNumElements(in, key) >= 0; }
int integer(const VSAPI* api, const VSMap* in, const char* key, int fallback, int low, int high, const char* ns) {
    if (!present(api, in, key)) return fallback;
    if (api->mapNumElements(in, key) != 1) fail_plain(ns, std::string(key) + " requires one integer");
    const int value = map_int(api, in, key, fallback);
    if (value < low || value > high) fail_plain(ns, std::string(key) + " outside supported range");
    return value;
}
double real(const VSAPI* api, const VSMap* in, const char* key, double fallback, double low, double high,
            const char* ns) {
    if (!present(api, in, key)) return fallback;
    if (api->mapNumElements(in, key) != 1) fail_plain(ns, std::string(key) + " requires one number");
    int error = 0;
    const double value = api->mapGetFloat(in, key, 0, &error);
    if (error || !std::isfinite(value) || value < low || value > high) fail_plain(ns, std::string("invalid ") + key);
    return value;
}
std::array<int, 2> stage(const VSAPI* api, const VSMap* in, const char* key, int fallback, int low, int high,
                         const char* ns) {
    std::array<int, 2> result{fallback, fallback};
    if (!present(api, in, key)) return result;
    const int count = api->mapNumElements(in, key);
    if (count < 1 || count > 2) fail(ns, "NLH", (std::string(key) + " must have one or two values").c_str());
    for (int i = 0; i < 2; ++i) {
        int error = 0;
        const auto value = api->mapGetInt(in, key, std::min(i, count - 1), &error);
        if (error || value < low || value > high) fail(ns, "NLH", (std::string("invalid ") + key).c_str());
        result[i] = int(value);
    }
    return result;
}

void parse_twsc(const VSAPI* api, const VSMap* in, bool sigma_given, FullImageParams& d, const char* ns) {
    auto& o = d.twsc;
    d.estimate = integer(api, in, "estimate_sigma", 0, 0, 1, ns) != 0;
    if (d.estimate && sigma_given) fail(ns, "TWSC", "estimate_sigma and explicit sigma are mutually exclusive");
    o.block = integer(api, in, "block_size", kTwscDefaultBlock, 1, 16, ns);
    o.group = integer(api, in, "group_size", kTwscDefaultGroup, 1, 256, ns);
    // The default step is independently overridable, but must remain valid
    // when a caller narrows the block without spelling a step. An explicitly
    // supplied step still goes through the range check and is rejected when
    // it exceeds the resolved block size.
    const bool step_given = present(api, in, "block_step");
    o.step = integer(api, in, "block_step", step_given ? kTwscDefaultStep : std::min(kTwscDefaultStep, o.block), 1,
                     o.block, ns);
    o.iterations = integer(api, in, "iters", kTwscDefaultIters, 1, 64, ns);
    o.window = present(api, in, "bm_range") ? 2 * integer(api, in, "bm_range", 0, 1, 64, ns) + 1
                                            : integer(api, in, "search_window", 60, 1, 129, ns);
    o.radius = integer(api, in, "radius", 0, 0, 16, ns);
    // Validate the API bound first, then report the model-specific
    // resolved-group violation below, keeping the documented
    // `ps_num exceeds resolved group_size` error reachable.
    o.ps_num = integer(api, in, "ps_num", 2, 1, 256, ns);
    o.ps_range = integer(api, in, "ps_range", 4, 1, 64, ns);
    if (o.group == 1 && !present(api, in, "ps_num")) o.ps_num = 1;
    if (o.ps_num > o.group) fail(ns, "TWSC", "ps_num exceeds resolved group_size");
    o.lambda2 = real(api, in, "lambda2", 1, 0, std::numeric_limits<float>::max(), ns);
    o.delta = real(api, in, "delta", 0, 0, 1, ns);
    o.solver.iterations = integer(api, in, "admm_iter", 10, 1, 1000, ns);
    o.solver.rho = real(api, in, "rho", 0.5, 0, std::numeric_limits<float>::max(), ns);
    o.solver.mu = real(api, in, "mu", 1.1, 1, std::numeric_limits<float>::max(), ns);
    o.solver.tolerance = real(api, in, "tol", 1e-6, 0, std::numeric_limits<float>::max(), ns);
    if (!(o.solver.rho > 0) || !(o.solver.tolerance > 0)) fail(ns, "TWSC", "rho and tol must be positive");
}

void parse_nlh(const VSAPI* api, const VSMap* in, const VSVideoInfo& vi, bool sigma_given, FullImageParams& d,
               const char* ns) {
    auto& o = d.nlh;
    d.estimate = !sigma_given;
    std::string noise_model = "auto";
    if (present(api, in, "noise_model")) {
        int e = 0;
        const char* value = api->mapGetData(in, "noise_model", 0, &e);
        if (e) fail(ns, "NLH", "invalid noise_model");
        noise_model.assign(value, std::size_t(api->mapGetDataSize(in, "noise_model", 0, nullptr)));
    }
    if (noise_model != "auto" && noise_model != "awgn" && noise_model != "real")
        fail(ns, "NLH", "noise_model must be auto, awgn, or real");
    o.real_noise = noise_model == "real" || (noise_model == "auto" && vi.format.colorFamily != cfGray);
    o.block = stage(api, in, "block_size", 0, 2, 16, ns);
    o.step = stage(api, in, "block_step", 0, 1, 16, ns);
    o.group = stage(api, in, "group_size", 0, 2, 64, ns);
    o.q = stage(api, in, "q", 0, 2, 16, ns);
    o.window = present(api, in, "bm_range")
                   ? std::array<int, 2>{2 * integer(api, in, "bm_range", 0, 1, 64, ns) + 1,
                                        2 * integer(api, in, "bm_range", 0, 1, 64, ns) + 1}
                   : stage(api, in, "search_window", 0, 1, 129, ns);
    for (int s = 0; s < 2; ++s)
        if ((o.group[s] & (o.group[s] - 1)) || (o.q[s] & (o.q[s] - 1)) ||
            (o.block[s] && (o.q[s] > o.block[s] * o.block[s] || o.step[s] > o.block[s])))
            fail(ns, "NLH", "invalid stage group/q/block/step combination");
    o.radius = integer(api, in, "radius", 0, 0, 16, ns);
    o.ps_num = integer(api, in, "ps_num", 2, 1, std::min(o.group[0] ? o.group[0] : 64, o.group[1] ? o.group[1] : 64), ns);
    o.ps_range = integer(api, in, "ps_range", 4, 1, 64, ns);
    o.basic_iterations = integer(api, in, "basic_iters", 0, 1, 64, ns);
    o.wiener_iterations = integer(api, in, "wiener_iters", 0, 1, 64, ns);
    o.basic_mix = real(api, in, "lambda_basic", -1, 0, 1, ns);
    o.hard_strength = real(api, in, "hard_strength", -1, 0, std::numeric_limits<float>::max(), ns);
    o.wiener_sigma_scale = real(api, in, "wiener_sigma_scale", -1, 0, std::numeric_limits<float>::max(), ns);
}

}  // namespace

FullImageParams parse_full_image(const VSAPI* api, const VSMap* in, const VSVideoInfo& vi,
                                 const VSVideoInfo* rclip_vi, Model model, const char* ns) {
    FullImageParams d;
    if (!is_const_32f(vi)) fail_plain(ns, "constant Gray/YUV/RGB 32-bit float required");
    const bool sigma_given = present(api, in, "sigma");
    if (sigma_given && (api->mapNumElements(in, "sigma") < 1 || api->mapNumElements(in, "sigma") > vi.format.numPlanes))
        fail_plain(ns, "sigma must contain one value per selected input plane (last value broadcasts)");
    map_float_array(api, in, "sigma", d.sigma.data(), vi.format.numPlanes, 3);
    for (int c = 0; c < vi.format.numPlanes; ++c) {
        if (sigma_given) d.sigma_units[c] = api->mapGetFloat(in, "sigma", std::min(c, api->mapNumElements(in, "sigma") - 1), nullptr);
        if (d.sigma_units[c] < 0) fail_plain(ns, "sigma must be nonnegative");
        if (d.sigma_units[c] > 0 && !(d.sigma[c] / 255.f > 0))
            fail_plain(ns, "positive sigma is not representable after float32 normalization");
    }
    if (rclip_vi && !same_video(vi, *rclip_vi)) fail_plain(ns, "rclip must match clip");
    if (present(api, in, "bm_range") && present(api, in, "search_window"))
        fail_plain(ns, "bm_range and search_window are mutually exclusive");
    if (model == Model::TWSC) parse_twsc(api, in, sigma_given, d, ns);
    else parse_nlh(api, in, vi, sigma_given, d, ns);
    return d;
}

void validate_full_image_geometry(const FullImageParams& p, const VSVideoInfo& vi, Model model, const char* ns) {
    const int planes = vi.format.numPlanes;
    if (p.estimate) {
        // Noise-estimated presets resolve per frame; their fixed bootstrap needs 8x8.
        for (int c = 0; c < planes; ++c)
            if (plane_width(vi, c) < 8 || plane_height(vi, c) < 8)
                fail_plain(ns, "noise estimation block_size requires at least 8x8 per plane");
        return;
    }
    std::array<float, 3> sigma{};
    std::array<double, 3> units{-1, -1, -1};
    for (int c = 0; c < planes; ++c) {
        sigma[c] = p.sigma[c] / 255.f;
        units[c] = p.sigma_units[c];
    }
    if (model == Model::NLH && vi.format.colorFamily == cfRGB) nlh_rgb_sigma_to_yuv(sigma, units);
    int blocks[2]{}, steps[2]{};
    int stages = 1;
    if (model == Model::TWSC) {
        blocks[0] = p.twsc.block;
        steps[0] = p.twsc.step;
    } else {
        double max_sigma = 0;
        int available = 16;
        for (int c = 0; c < planes; ++c) {
            const double u = units[c] >= 0 ? units[c] : double(sigma[c]) * 255;
            if (!(u >= 0) || !std::isfinite(u)) fail(ns, "NLH", "invalid preset sigma");
            max_sigma = std::max(max_sigma, u);
            if (sigma[c] > 0) available = std::min({available, plane_width(vi, c), plane_height(vi, c)});
        }
        const auto o = nlh_resolve_preset(max_sigma, available, p.nlh);
        stages = 2;
        for (int s = 0; s < 2; ++s) {
            blocks[s] = o.block[s];
            steps[s] = o.step[s];
        }
    }
    for (int s = 0; s < stages; ++s) {
        if (steps[s] > blocks[s]) fail_plain(ns, "block_step exceeds resolved block_size");
        for (int c = 0; c < planes; ++c)
            if (sigma[c] > 0 && (plane_width(vi, c) < blocks[s] || plane_height(vi, c) < blocks[s]))
                fail_plain(ns, "selected plane is smaller than block_size");
    }
}

}  // namespace nss::frontend
