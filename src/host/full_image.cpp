#include "host/full_image.hpp"
#include "host/validate.hpp"
#include "host/temporal.hpp"
#include "host/contribution.hpp"
#include "nss/backend.hpp"
#include "nss/cpu_image.hpp"
#include "nss/cpu_nlh_full.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>

namespace {
struct FullImageData {
    std::shared_ptr<nss::ResourceBudget> budget = nss::current_budget();
    nss::NodeRef node, reference;
    VSVideoInfo vi{}, output{};
    nss::Model model = nss::Model::TWSC;
    nss::TwscImageOptions twsc;
    nss::NlhImageOptions nlh;
    std::array<float, 3> sigma{3, 3, 3};
    std::array<double, 3> sigma_units{3, 3, 3};
    bool estimate = false;
    int radius() const { return model == nss::Model::TWSC ? twsc.radius : nlh.radius; }
};
bool present(const VSAPI* api, const VSMap* in, const char* key) { return api->mapNumElements(in, key) >= 0; }
int integer(const VSAPI* api, const VSMap* in, const char* key, int fallback, int low, int high) {
    if (!present(api, in, key)) return fallback;
    if (api->mapNumElements(in, key) != 1) throw std::invalid_argument(std::string("nss: ") + key + " requires one integer");
    const int value = nss::map_int(api, in, key, fallback);
    if (value < low || value > high) throw std::invalid_argument(std::string("nss: ") + key + " outside supported range");
    return value;
}
double real(const VSAPI* api, const VSMap* in, const char* key, double fallback, double low, double high) {
    if (!present(api, in, key)) return fallback;
    if (api->mapNumElements(in, key) != 1) throw std::invalid_argument(std::string("nss: ") + key + " requires one number");
    int error = 0;
    const double value = api->mapGetFloat(in, key, 0, &error);
    if (error || !std::isfinite(value) || value < low || value > high) throw std::invalid_argument(std::string("nss: invalid ") + key);
    return value;
}
std::array<int, 2> stage(const VSAPI* api, const VSMap* in, const char* key, int fallback, int low, int high) {
    std::array<int, 2> result{fallback, fallback};
    if (!present(api, in, key)) return result;
    const int count = api->mapNumElements(in, key);
    if (count < 1 || count > 2) throw std::invalid_argument(std::string("nss.NLH: ") + key + " must have one or two values");
    for (int i = 0; i < 2; ++i) {
        int error = 0;
        const auto value = api->mapGetInt(in, key, std::min(i, count - 1), &error);
        if (error || value < low || value > high) throw std::invalid_argument(std::string("nss.NLH: invalid ") + key);
        result[i] = int(value);
    }
    return result;
}
void resolve_twsc(const FullImageData& d, const nss::ImageFrame& frame, int* block, int* group, int* iters) {
    double total = 0; int active = 0;
    for (int c = 0; c < d.vi.format.numPlanes; ++c) if (frame.sigma[c] > 0) { const double s = nss::image_sigma_units(frame,c); total += s * s; ++active; }
    const double sigma = active ? std::sqrt(total / active) : 0;
    block[0] = d.twsc.block ? d.twsc.block : sigma <= 20 ? 7 : sigma <= 60 ? 8 : 9;
    group[0] = d.twsc.group ? d.twsc.group : sigma <= 20 ? 70 : sigma <= 40 ? 90 : sigma <= 60 ? 120 : 140;
    iters[0] = d.twsc.iterations ? d.twsc.iterations : sigma <= 20 ? 8 : sigma <= 60 ? 12 : 14;
}
void diagnostic_ints(const VSAPI* api, VSMap* props, const char* key, const int* values, int count) {
    std::int64_t data[2]{values[0], count > 1 ? values[1] : 0};
    if (api->mapSetIntArray(props, key, data, count)) throw std::bad_alloc();
}
const VSFrame* VS_CC fullGetFrame(int n, int activation, void* instance, void**, VSFrameContext* context,
                                 VSCore* core, const VSAPI* api) {
    auto& d = *static_cast<FullImageData*>(instance);
    nss::ResourceScope scope(d.budget);
    const int radius = d.radius();
    const int first = nss::host_detail::temporal_first(n, radius);
    const int last = nss::host_detail::temporal_last(n, radius, d.vi.numFrames);
    if (activation == arInitial) {
        for (int frame = first; frame <= last; ++frame) {
            api->requestFrameFilter(frame, d.node, context);
            if (d.reference) api->requestFrameFilter(frame, d.reference, context);
        }
        return nullptr;
    }
    if (activation != arAllFramesReady) return nullptr;
    nss::FrameScope owned(api);
    const int count = last - first + 1, center = n - first, planes = d.vi.format.numPlanes;
    nss::ResourceVector<const VSFrame*> source(count);
    nss::ImageSequence input(count), reference(d.reference ? count : 0);
    auto load = [&](const VSFrame* frame, nss::ImageFrame& image) {
        for (int c = 0; c < planes; ++c) {
            const int width = nss::plane_width(d.vi, c), height = nss::plane_height(d.vi, c);
            const int stride = int(owned.getStride(frame, c) / sizeof(float));
            auto& p = image.planes[c]; p = nss::ImagePlane(width, height);
            const float* pixels = reinterpret_cast<const float*>(api->getReadPtr(frame, c));
            for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
                const float value = pixels[y * stride + x];
                if (!std::isfinite(value)) throw std::invalid_argument("nss: nonfinite source/reference sample");
                p.pixels[y * width + x] = value;
            }
            image.sigma[c] = d.sigma[c] / 255.f;
            image.sigma_units[c] = d.sigma_units[c];
        }
    };
    for (int t = 0; t < count; ++t) {
        source[t] = owned.getFrameFilter(first + t, d.node, context);
        load(source[t], input[t]);
        if (d.reference) {
            const auto* frame = owned.getFrameFilter(first + t, d.reference, context);
            load(frame, reference[t]); owned.freeFrame(frame);
        }
    }
    const bool rgb = d.model == nss::Model::NLH && d.vi.format.colorFamily == cfRGB;
    if (rgb) for (int t = 0; t < count; ++t) {
        nss::nlh_rgb_to_yuv(input[t], !d.estimate);
        if (d.reference) nss::nlh_rgb_to_yuv(reference[t], false);
    }
    if (d.estimate) for (int t = 0; t < count; ++t) {
        const auto& matching = d.reference ? reference[t] : input[t];
        if (d.model == nss::Model::NLH) nss::nlh_estimate_frame_sigma(input[t], matching, planes);
        else for (int c = 0; c < planes; ++c) {
            input[t].sigma[c] = nss::nlh_estimate_sigma(input[t].planes[c], matching.planes[c]);
            input[t].sigma_units[c] = double(input[t].sigma[c]) * 255;
        }
    }
    bool bypass = true;
    for (int c = 0; c < planes; ++c) bypass = bypass && input[center].sigma[c] == 0;
    nss::NlhImageOptions nlh_options;
    int blocks[2]{}, groups[2]{}, iterations[2]{};
    if (d.model == nss::Model::NLH) {
        // A bypassed center does not filter its temporal neighbors. Resolve
        // diagnostics from the center alone, retaining zero-noise identity.
        nlh_options = bypass
            ? nss::nlh_resolve_options({&input[center], 1}, planes, 0, d.nlh)
            : nss::nlh_resolve_options(input, planes, center, d.nlh);
        std::copy_n(nlh_options.block.data(), 2, blocks);
        std::copy_n(nlh_options.group.data(), 2, groups);
        iterations[0] = nlh_options.basic_iterations;
        iterations[1] = nlh_options.wiener_iterations;
    } else resolve_twsc(d, input[center], blocks, groups, iterations);
    auto* dst = owned.newVideoFrame(&d.output.format, d.output.width, d.output.height, source[center], core);
    nss::stamp_contribution(dst, radius, n, d.model, api);
    nss::ImageContributions result;
    if (!bypass) {
        result = d.model == nss::Model::TWSC
            ? nss::twsc_image(input, d.reference ? &reference : nullptr, planes, center, d.twsc)
            : nss::nlh_image(input, d.reference ? &reference : nullptr, planes, center, nlh_options);
        if (rgb) for (int t = 0; t < count; ++t) {
            // Shared pixel indices give all three channels identical counts.
            if (result.denominator[t].planes[0].pixels != result.denominator[t].planes[1].pixels ||
                result.denominator[t].planes[0].pixels != result.denominator[t].planes[2].pixels)
                throw std::logic_error("nss.NLH: inconsistent shared color aggregation");
            nss::nlh_yuv_to_rgb(result.numerator[t]);
        }
    }
    for (int c = 0; c < planes; ++c) {
        const int width = nss::plane_width(d.vi, c), height = nss::plane_height(d.vi, c);
        const int stride = int(owned.getStride(dst, c) / sizeof(float));
        const int src_stride = int(owned.getStride(source[center], c) / sizeof(float));
        float* out = reinterpret_cast<float*>(api->getWritePtr(dst, c));
        const float* original = reinterpret_cast<const float*>(api->getReadPtr(source[center], c));
        const bool selected_off = !d.estimate && d.sigma[c] == 0;
        if (bypass || selected_off) {
            if (radius) nss::host_detail::temporal_identity(out, stride, original, src_stride, width, height, radius);
            else for (int y = 0; y < height; ++y) std::memcpy(out + y * stride, original + y * src_stride, std::size_t(width) * sizeof(float));
            continue;
        }
        if (radius) {
            for (int sl = 0; sl < 2 * radius + 1; ++sl) for (int y = 0; y < height; ++y) {
                float* num = out + (std::size_t(2 * sl) * height + y) * stride;
                float* den = num + std::size_t(height) * stride;
                const int t = n - radius + sl - first;
                if (t < 0 || t >= count) { std::fill_n(num, width, 0); std::fill_n(den, width, 0); }
                else {
                    std::copy_n(result.numerator[t].planes[c].pixels.data() + y * width, width, num);
                    std::copy_n(result.denominator[t].planes[c].pixels.data() + y * width, width, den);
                }
            }
        } else {
            const auto& num = result.numerator[center].planes[c].pixels;
            const auto& den = result.denominator[center].planes[c].pixels;
            for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
                const auto index = y * width + x;
                const double value = den[index] > 0 ? double(num[index]) / den[index] : original[y * src_stride + x];
                if (!std::isfinite(value)) throw std::runtime_error("nss: nonfinite final estimate");
                out[y * stride + x] = float(value);
            }
        }
    }
    auto* props = api->getFramePropertiesRW(dst);
    double sigmas[3]{};
    for (int c = 0; c < planes; ++c) sigmas[c] = nss::image_sigma_units(input[center],c);
    if (api->mapSetFloatArray(props, "_NSSSigma", sigmas, planes) ||
        api->mapSetInt(props, "_NSSGroups", std::int64_t(result.stats.groups), maReplace) ||
        api->mapSetInt(props, "_NSSADMMMaxIterGroups", std::int64_t(result.stats.max_iteration_groups), maReplace) ||
        api->mapSetInt(props, "_NSSSvdDoubleGroups", std::int64_t(result.stats.double_svd_groups), maReplace) ||
        api->mapSetFloat(props, "_NSSSylvesterResidual", result.stats.max_sylvester_residual, maReplace)) throw std::bad_alloc();
    const int stages = d.model == nss::Model::TWSC ? 1 : 2;
    diagnostic_ints(api, props, "_NSSBlockSize", blocks, stages);
    diagnostic_ints(api, props, "_NSSGroupSize", groups, stages);
    diagnostic_ints(api, props, "_NSSIterations", iterations, stages);
    const int windows[2]{d.model == nss::Model::TWSC ? d.twsc.window : nlh_options.window[0], nlh_options.window[1]};
    const int steps[2]{d.model == nss::Model::TWSC ? d.twsc.step : nlh_options.step[0], nlh_options.step[1]};
    diagnostic_ints(api, props, "_NSSSearchWindow", windows, stages);
    diagnostic_ints(api, props, "_NSSBlockStep", steps, stages);
    if (d.model == nss::Model::NLH) {
        diagnostic_ints(api, props, "_NSSQ", nlh_options.q.data(), 2);
        if (api->mapSetFloat(props, "_NSSLambdaBasic", nlh_options.basic_mix, maReplace) ||
            api->mapSetFloat(props, "_NSSHardStrength", nlh_options.hard_strength, maReplace) ||
            api->mapSetFloat(props, "_NSSHardCoefficient", nss::kNlhHardCoefficient * nlh_options.hard_strength, maReplace) ||
            api->mapSetFloat(props, "_NSSWienerSigmaScale", nlh_options.wiener_sigma_scale, maReplace)) throw std::bad_alloc();
    }
    return owned.keep(dst);
}
void VS_CC fullFree(void* instance, VSCore*, const VSAPI*) { delete static_cast<FullImageData*>(instance); }
} // namespace

VSNode* nss_create_full_image(const VSMap* in, VSCore* core, const VSAPI* api, VSMap* error, nss::Model model) {
    if (!nss::cpu_backend_available()) { api->mapSetError(error, "nss: no executable CPU backend"); return nullptr; }
    auto data = std::make_unique<FullImageData>(); auto& d = *data;
    d.model = model; d.node = nss::get_node(api, in, "clip", 0, nullptr);
    d.vi = *api->getVideoInfo(d.node);
    if (!nss::is_const_32f(d.vi)) throw std::invalid_argument("nss: constant Gray/YUV/RGB 32-bit float required");
    const bool sigma_given = present(api, in, "sigma");
    if (sigma_given && (api->mapNumElements(in, "sigma") < 1 || api->mapNumElements(in, "sigma") > d.vi.format.numPlanes))
        throw std::invalid_argument("nss: sigma must contain one value per selected input plane (last value broadcasts)");
    nss::map_float_array(api, in, "sigma", d.sigma.data(), d.vi.format.numPlanes, 3);
    for (int c = 0; c < d.vi.format.numPlanes; ++c) {
        if (sigma_given) d.sigma_units[c] = api->mapGetFloat(in, "sigma", std::min(c, api->mapNumElements(in,"sigma")-1), nullptr);
        if (d.sigma_units[c] < 0) throw std::invalid_argument("nss: sigma must be nonnegative");
        if (d.sigma_units[c] > 0 && !(d.sigma[c] / 255.f > 0))
            throw std::invalid_argument("nss: positive sigma is not representable after float32 normalization");
    }
    if (present(api, in, "rclip")) {
        d.reference = nss::get_node(api, in, "rclip", 0, nullptr);
        if (!nss::same_video(d.vi, *api->getVideoInfo(d.reference))) throw std::invalid_argument("nss: rclip must match clip");
    }
    if (present(api, in, "bm_range") && present(api, in, "search_window")) throw std::invalid_argument("nss: bm_range and search_window are mutually exclusive");
    if (model == nss::Model::TWSC) {
        auto& o = d.twsc;
        d.estimate = integer(api, in, "estimate_sigma", 0, 0, 1) != 0;
        if (d.estimate && sigma_given) throw std::invalid_argument("nss.TWSC: estimate_sigma and explicit sigma are mutually exclusive");
        o.block = integer(api, in, "block_size", 0, 1, 16);
        o.group = integer(api, in, "group_size", 0, 1, 256);
        o.step = integer(api, in, "block_step", 1, 1, o.block ? o.block : 16);
        o.iterations = integer(api, in, "iters", 0, 1, 64);
        o.window = present(api, in, "bm_range") ? 2 * integer(api, in, "bm_range", 0, 1, 64) + 1 : integer(api, in, "search_window", 60, 1, 129);
        o.radius = integer(api, in, "radius", 0, 0, 16);
        o.ps_num = integer(api, in, "ps_num", 2, 1, o.group ? o.group : 256);
        o.ps_range = integer(api, in, "ps_range", 4, 1, 64);
        if (o.group == 1 && !present(api, in, "ps_num")) o.ps_num = 1;
        o.lambda2 = real(api, in, "lambda2", 1, 0, std::numeric_limits<float>::max());
        o.delta = real(api, in, "delta", 0, 0, 1);
        o.solver.iterations = integer(api, in, "admm_iter", 10, 1, 1000);
        o.solver.rho = real(api, in, "rho", 0.5, 0, std::numeric_limits<float>::max());
        o.solver.mu = real(api, in, "mu", 1.1, 1, std::numeric_limits<float>::max());
        o.solver.tolerance = real(api, in, "tol", 1e-6, 0, std::numeric_limits<float>::max());
        if (!(o.solver.rho > 0) || !(o.solver.tolerance > 0)) throw std::invalid_argument("nss.TWSC: rho and tol must be positive");
    } else {
        auto& o = d.nlh;
        d.estimate = !sigma_given;
        std::string noise_model = "auto";
        if (present(api, in, "noise_model")) {
            int e = 0; const char* value = api->mapGetData(in, "noise_model", 0, &e);
            if (e) throw std::invalid_argument("nss.NLH: invalid noise_model");
            noise_model.assign(value, std::size_t(api->mapGetDataSize(in, "noise_model", 0, nullptr)));
        }
        if (noise_model != "auto" && noise_model != "awgn" && noise_model != "real") throw std::invalid_argument("nss.NLH: noise_model must be auto, awgn, or real");
        o.real_noise = noise_model == "real" || (noise_model == "auto" && d.vi.format.colorFamily != cfGray);
        o.block = stage(api, in, "block_size", 0, 2, 16); o.step = stage(api, in, "block_step", 0, 1, 16);
        o.group = stage(api, in, "group_size", 0, 2, 64); o.q = stage(api, in, "q", 0, 2, 16);
        o.window = present(api, in, "bm_range") ? std::array<int, 2>{2 * integer(api, in, "bm_range", 0, 1, 64) + 1, 2 * integer(api, in, "bm_range", 0, 1, 64) + 1} : stage(api, in, "search_window", 0, 1, 129);
        for (int s = 0; s < 2; ++s) if ((o.group[s] & (o.group[s] - 1)) || (o.q[s] & (o.q[s] - 1)) ||
            (o.block[s] && (o.q[s] > o.block[s] * o.block[s] || o.step[s] > o.block[s])))
            throw std::invalid_argument("nss.NLH: invalid stage group/q/block/step combination");
        o.radius = integer(api, in, "radius", 0, 0, 16);
        o.ps_num = integer(api, in, "ps_num", 2, 1, std::min(o.group[0] ? o.group[0] : 64, o.group[1] ? o.group[1] : 64));
        o.ps_range = integer(api, in, "ps_range", 4, 1, 64);
        o.basic_iterations = integer(api, in, "basic_iters", 0, 1, 64); o.wiener_iterations = integer(api, in, "wiener_iters", 0, 1, 64);
        o.basic_mix = real(api, in, "lambda_basic", -1, 0, 1); o.hard_strength = real(api, in, "hard_strength", -1, 0, std::numeric_limits<float>::max());
        o.wiener_sigma_scale = real(api, in, "wiener_sigma_scale", -1, 0, std::numeric_limits<float>::max());
    }
    // Known geometry errors remain creation-time errors. Noise-estimated
    // presets are resolved per frame, but their fixed bootstrap needs 8x8.
    if (d.estimate) {
        for (int c = 0; c < d.vi.format.numPlanes; ++c)
            if (nss::plane_width(d.vi, c) < 8 || nss::plane_height(d.vi, c) < 8)
                throw std::invalid_argument("nss: noise estimation block_size requires at least 8x8 per plane");
    } else {
        nss::ImageFrame metadata;
        for (int c = 0; c < d.vi.format.numPlanes; ++c) {
            metadata.planes[c].width = nss::plane_width(d.vi, c);
            metadata.planes[c].height = nss::plane_height(d.vi, c);
            metadata.sigma[c] = d.sigma[c] / 255.f;
            metadata.sigma_units[c] = d.sigma_units[c];
        }
        if (model == nss::Model::NLH && d.vi.format.colorFamily == cfRGB)
            nss::nlh_rgb_to_yuv(metadata, true);
        int blocks[2]{}, groups[2]{}, iterations[2]{};
        nss::NlhImageOptions nlh_options;
        if (model == nss::Model::TWSC) resolve_twsc(d, metadata, blocks, groups, iterations);
        else {
            nlh_options = nss::nlh_resolve_options({&metadata, 1}, d.vi.format.numPlanes, 0, d.nlh);
            std::copy_n(nlh_options.block.data(), 2, blocks);
            std::copy_n(nlh_options.group.data(), 2, groups);
        }
        const int stages = model == nss::Model::TWSC ? 1 : 2;
        for (int s = 0; s < stages; ++s) {
            const int step = model == nss::Model::TWSC ? d.twsc.step : nlh_options.step[s];
            if (step > blocks[s]) throw std::invalid_argument("nss: block_step exceeds resolved block_size");
            for (int c = 0; c < d.vi.format.numPlanes; ++c)
                if (metadata.sigma[c] > 0 && (nss::plane_width(d.vi,c) < blocks[s] || nss::plane_height(d.vi,c) < blocks[s]))
                    throw std::invalid_argument("nss: selected plane is smaller than block_size");
        }
    }
    d.output = d.vi;
    if (d.radius()) d.output.height = nss::checked_fat_height(d.vi.height, d.radius());
    VSFilterDependency deps[2]{{d.node, d.radius() ? rpGeneral : rpStrictSpatial}, {d.reference, d.radius() ? rpGeneral : rpStrictSpatial}};
    VSNode* node = api->createVideoFilter2(model == nss::Model::TWSC ? "TWSC" : "NLH", &d.output,
                                          nss::checked_frame<fullGetFrame>, fullFree, fmParallel, deps, d.reference ? 2 : 1, &d, core);
    if (!node) { api->mapSetError(error, "nss: failed to create full image filter"); return nullptr; }
    data.release(); return node;
}
