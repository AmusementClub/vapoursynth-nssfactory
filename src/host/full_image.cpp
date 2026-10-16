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
#include <stdexcept>
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
// A frame's planes as an image (nonfinite samples are an error), with the
// given sigma of each plane.
void load_image(const FullImageData& d, nss::FrameScope& owned, const VSAPI* api, const VSFrame* frame,
                nss::ImageFrame& image) {
    for (int c = 0; c < d.vi.format.numPlanes; ++c) {
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
}

// The window's frames in the model's working domain (TWSC cfast: three
// opponent planes; NLH RGB: YUV), with their sigma estimated when blind.
void prepare_window(const FullImageData& d, nss::ImageSequence& input, nss::ImageSequence& reference, bool split,
                    bool rgb) {
    const int count = int(input.size()), planes = d.vi.format.numPlanes;
    const auto opponent = [](nss::ImageFrame& image) {
        auto& p = image.planes;
        for (std::size_t i = 0; i < p[0].pixels.size(); ++i)
            nss::twsc_opponent_forward(p[0].pixels[i], p[1].pixels[i], p[2].pixels[i]);
    };
    if (split) for (int t = 0; t < count; ++t) {
        opponent(input[t]);
        if (d.reference) opponent(reference[t]);
    }
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
}

// The shapes a frame runs with, per stage (TWSC has one), for its diagnostics.
struct Shapes {
    nss::NlhImageOptions nlh;  // NLH: the options resolved for this frame
    int blocks[2]{}, groups[2]{}, iterations[2]{}, windows[2]{}, steps[2]{};
    int stages = 1;
};

Shapes resolve_shapes(const FullImageData& d, const nss::ImageSequence& input, int center, bool bypass) {
    const int planes = d.vi.format.numPlanes;
    Shapes s;
    if (d.model == nss::Model::NLH) {
        // A bypassed center does not filter its temporal neighbors. Resolve
        // diagnostics from the center alone, retaining zero-noise identity.
        s.nlh = bypass
            ? nss::nlh_resolve_options({&input[center], 1}, planes, 0, d.nlh)
            : nss::nlh_resolve_options(input, planes, center, d.nlh);
        std::copy_n(s.nlh.block.data(), 2, s.blocks);
        std::copy_n(s.nlh.group.data(), 2, s.groups);
        s.iterations[0] = s.nlh.basic_iterations;
        s.iterations[1] = s.nlh.wiener_iterations;
        std::copy_n(s.nlh.window.data(), 2, s.windows);
        std::copy_n(s.nlh.step.data(), 2, s.steps);
        s.stages = 2;
    } else {
        s.blocks[0] = d.twsc.block;
        s.groups[0] = d.twsc.group;
        s.iterations[0] = d.twsc.iterations;
        s.windows[0] = d.twsc.window;
        s.steps[0] = d.twsc.step;
    }
    return s;
}

// The final value of a pixel: its estimate, or its input where no group reached.
float finished(float num, float den, float input) {
    const double value = den > 0 ? double(num) / den : input;
    if (!std::isfinite(value)) throw std::runtime_error("nss: nonfinite final estimate");
    return float(value);
}

// cfast: each opponent plane's estimate, taken back to RGB into dst.
void write_split(const FullImageData& d, nss::FrameScope& owned, const VSAPI* api, VSFrame* dst,
                 const nss::ImageContributions& result, const nss::ImageFrame& input, int center) {
    const int width = d.vi.width, height = d.vi.height;
    float* out[3]; int stride[3];
    for (int c = 0; c < 3; ++c) {
        out[c] = reinterpret_cast<float*>(api->getWritePtr(dst, c));
        stride[c] = int(owned.getStride(dst, c) / sizeof(float));
    }
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        const auto index = y * width + x;
        float value[3];
        for (int c = 0; c < 3; ++c) {
            value[c] = finished(result.numerator[center].planes[c].pixels[index],
                                result.denominator[center].planes[c].pixels[index], input.planes[c].pixels[index]);
        }
        nss::twsc_opponent_inverse(value[0], value[1], value[2]);
        for (int c = 0; c < 3; ++c) out[c][y * stride[c] + x] = value[c];
    }
}

// Plane c of dst: the source when the plane is not filtered (its identity
// slices when temporal), the weighted slices of the window (temporal), or the
// finished estimate.
void write_plane(const FullImageData& d, nss::FrameScope& owned, const VSAPI* api, VSFrame* dst, int c,
                 const VSFrame* source, const nss::ImageContributions& result, bool filtered, int n, int first,
                 int count) {
    const int radius = d.radius(), center = n - first;
    const int width = nss::plane_width(d.vi, c), height = nss::plane_height(d.vi, c);
    const int stride = int(owned.getStride(dst, c) / sizeof(float));
    const int src_stride = int(owned.getStride(source, c) / sizeof(float));
    float* out = reinterpret_cast<float*>(api->getWritePtr(dst, c));
    const float* original = reinterpret_cast<const float*>(api->getReadPtr(source, c));
    if (!filtered) {
        if (radius) nss::host_detail::temporal_identity(out, stride, original, src_stride, width, height, radius);
        else for (int y = 0; y < height; ++y) std::memcpy(out + y * stride, original + y * src_stride, std::size_t(width) * sizeof(float));
        return;
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
        return;
    }
    const auto& num = result.numerator[center].planes[c].pixels;
    const auto& den = result.denominator[center].planes[c].pixels;
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        const auto index = y * width + x;
        out[y * stride + x] = finished(num[index], den[index], original[y * src_stride + x]);
    }
}

// The resolved values and solver statistics as _NSS* frame properties.
void stamp_diagnostics(const FullImageData& d, const VSAPI* api, VSFrame* dst, const nss::ImageFrame& center,
                       const nss::ImageContributions& result, const Shapes& shapes, bool split) {
    const int planes = d.vi.format.numPlanes;
    auto* props = api->getFramePropertiesRW(dst);
    double sigmas[3]{};
    for (int c = 0; c < planes; ++c) sigmas[c] = nss::image_sigma_units(center, c);
    if (api->mapSetFloatArray(props, "_NSSSigma", sigmas, planes) ||
        api->mapSetInt(props, "_NSSGroups", std::int64_t(result.stats.groups), maReplace) ||
        api->mapSetInt(props, "_NSSADMMMaxIterGroups", std::int64_t(result.stats.max_iteration_groups), maReplace) ||
        api->mapSetInt(props, "_NSSSvdDoubleGroups", std::int64_t(result.stats.double_svd_groups), maReplace) ||
        api->mapSetFloat(props, "_NSSSylvesterResidual", result.stats.max_sylvester_residual, maReplace)) throw std::bad_alloc();
    if (d.model == nss::Model::TWSC && api->mapSetInt(props, "_NSSCFast", split, maReplace)) throw std::bad_alloc();
    diagnostic_ints(api, props, "_NSSBlockSize", shapes.blocks, shapes.stages);
    diagnostic_ints(api, props, "_NSSGroupSize", shapes.groups, shapes.stages);
    diagnostic_ints(api, props, "_NSSIterations", shapes.iterations, shapes.stages);
    diagnostic_ints(api, props, "_NSSSearchWindow", shapes.windows, shapes.stages);
    diagnostic_ints(api, props, "_NSSBlockStep", shapes.steps, shapes.stages);
    if (d.model == nss::Model::NLH) {
        const auto& nlh = shapes.nlh;
        diagnostic_ints(api, props, "_NSSQ", nlh.q.data(), 2);
        if (api->mapSetFloat(props, "_NSSLambdaBasic", nlh.basic_mix, maReplace) ||
            api->mapSetFloat(props, "_NSSHardStrength", nlh.hard_strength, maReplace) ||
            api->mapSetFloat(props, "_NSSHardCoefficient", nss::kNlhHardCoefficient * nlh.hard_strength, maReplace) ||
            api->mapSetFloat(props, "_NSSWienerSigmaScale", nlh.wiener_sigma_scale, maReplace)) throw std::bad_alloc();
    }
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
    for (int t = 0; t < count; ++t) {
        source[t] = owned.getFrameFilter(first + t, d.node, context);
        load_image(d, owned, api, source[t], input[t]);
        if (d.reference) {
            const auto* frame = owned.getFrameFilter(first + t, d.reference, context);
            load_image(d, owned, api, frame, reference[t]); owned.freeFrame(frame);
        }
    }
    const bool rgb = d.model == nss::Model::NLH && d.vi.format.colorFamily == cfRGB;
    const bool split = d.model == nss::Model::TWSC &&
        nss::twsc_color_split(d.twsc, d.vi.format.colorFamily == cfRGB, d.estimate, d.sigma_units.data());
    prepare_window(d, input, reference, split, rgb);
    bool bypass = true;
    for (int c = 0; c < planes; ++c) bypass = bypass && input[center].sigma[c] == 0;
    const Shapes shapes = resolve_shapes(d, input, center, bypass);
    auto* dst = owned.newVideoFrame(&d.output.format, d.output.width, d.output.height, source[center], core);
    nss::stamp_contribution(dst, radius, n, d.model, api);
    nss::ImageContributions result;
    if (!bypass) {
        nss::TwscImageOptions twsc_options = d.twsc;
        twsc_options.planes_alone = split;
        result = d.model == nss::Model::TWSC
            ? nss::twsc_image(input, d.reference ? &reference : nullptr, planes, center, twsc_options)
            : nss::nlh_image(input, d.reference ? &reference : nullptr, planes, center, shapes.nlh);
        if (rgb) for (int t = 0; t < count; ++t) {
            // Shared pixel indices give all three channels identical counts.
            if (result.denominator[t].planes[0].pixels != result.denominator[t].planes[1].pixels ||
                result.denominator[t].planes[0].pixels != result.denominator[t].planes[2].pixels)
                throw std::logic_error("nss.NLH: inconsistent shared color aggregation");
            nss::nlh_yuv_to_rgb(result.numerator[t]);
        }
    }
    if (split && !bypass) {
        write_split(d, owned, api, dst, result, input[center], center);
    } else {
        for (int c = 0; c < planes; ++c) {
            const bool filtered = !bypass && (d.estimate || d.sigma[c] != 0);
            write_plane(d, owned, api, dst, c, source[center], result, filtered, n, first, count);
        }
    }
    stamp_diagnostics(d, api, dst, input[center], result, shapes, split);
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
