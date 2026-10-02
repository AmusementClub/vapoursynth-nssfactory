// SPDX-License-Identifier: GPL-2.0-only
#include "host/full_image.hpp"
#include "frontend/validate.hpp"
#include "frontend/temporal.hpp"
#include "frontend/contribution.hpp"
#include "frontend/full_image_args.hpp"
#include "nss/backend.hpp"
#include "nss/cpu_image.hpp"
#include "nss/cpu_nlh_full.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>

namespace {
struct FullImageData : nss::FullImageParams {
    std::shared_ptr<nss::ResourceBudget> budget = nss::current_budget();
    nss::NodeRef node, reference;
    VSVideoInfo vi{}, output{};
    nss::Model model = nss::Model::TWSC;
    int radius() const { return model == nss::Model::TWSC ? twsc.radius : nlh.radius; }
};
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
    } else {
        blocks[0] = d.twsc.block;
        groups[0] = d.twsc.group;
        iterations[0] = d.twsc.iterations;
    }
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
    if (api->mapNumElements(in, "rclip") >= 0) d.reference = nss::get_node(api, in, "rclip", 0, nullptr);
    static_cast<nss::FullImageParams&>(d) = nss::frontend::parse_full_image(
        api, in, d.vi, d.reference ? api->getVideoInfo(d.reference) : nullptr, model, "nss");
    nss::frontend::validate_full_image_geometry(d, d.vi, model, "nss");
    d.output = d.vi;
    if (d.radius()) d.output.height = nss::checked_fat_height(d.vi.height, d.radius());
    VSFilterDependency deps[2]{{d.node, d.radius() ? rpGeneral : rpStrictSpatial}, {d.reference, d.radius() ? rpGeneral : rpStrictSpatial}};
    VSNode* node = api->createVideoFilter2(model == nss::Model::TWSC ? "TWSC" : "NLH", &d.output,
                                          nss::checked_frame<fullGetFrame>, fullFree, fmParallel, deps, d.reference ? 2 : 1, &d, core);
    if (!node) { api->mapSetError(error, "nss: failed to create full image filter"); return nullptr; }
    data.release(); return node;
}
