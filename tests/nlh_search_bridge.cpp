// Experiment-only adapter. Reuse the production image API, caching only input
// conversion and noise estimation, which are independent of the searched knobs.
#include "nss/cpu_image.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
thread_local std::string last_error;
struct Source {
    int width, height, planes;
    nss::ImageSequence input;
    std::vector<float> original;
    double prepare_seconds;
};
struct Parameters {
    int block[2], step[2], group[2], q[2], window[2];
    int basic_iterations, wiener_iterations, real_noise;
    double mix, hard_strength, wiener_scale;
};
}

extern "C" {
const char* nlh_search_error() { return last_error.c_str(); }
std::size_t nlh_search_parameters_size() { return sizeof(Parameters); }

void* nlh_search_prepare(const float* pixels, int width, int height, int planes,
                         double sigma, int estimate) {
    try {
        const auto start = Clock::now();
        if (!pixels || width < 2 || height < 2 || (planes != 1 && planes != 3) ||
            (!estimate && (!(sigma >= 0) || !std::isfinite(sigma))))
            throw std::invalid_argument("invalid native search source");
        auto source = std::make_unique<Source>();
        source->width = width; source->height = height; source->planes = planes;
        const std::size_t count = std::size_t(width) * height;
        source->original.assign(pixels, pixels + count * planes);
        source->input.resize(1);
        auto& image = source->input[0];
        for (int c = 0; c < planes; ++c) {
            image.planes[c] = nss::ImagePlane(width, height);
            for (std::size_t i = 0; i < count; ++i) {
                const float value = pixels[c * count + i];
                if (!std::isfinite(value)) throw std::invalid_argument("nonfinite search input");
                image.planes[c].pixels[i] = value;
            }
            image.sigma[c] = estimate ? 0 : float(sigma) / 255.f;
            image.sigma_units[c] = estimate ? 0 : sigma;
        }
        if (planes == 3) nss::nlh_rgb_to_yuv(image, !estimate);
        if (estimate) nss::nlh_estimate_frame_sigma(image, image, planes);
        source->prepare_seconds = std::chrono::duration<double>(Clock::now() - start).count();
        return source.release();
    } catch (const std::exception& error) { last_error = error.what(); return nullptr; }
}

void nlh_search_free(void* handle) { delete static_cast<Source*>(handle); }
double nlh_search_prepare_seconds(const void* handle) {
    return static_cast<const Source*>(handle)->prepare_seconds;
}
void nlh_search_sigma(const void* handle, double* sigma) {
    const auto& source = *static_cast<const Source*>(handle);
    for (int c = 0; c < source.planes; ++c) sigma[c] = source.input[0].sigma_units[c];
}

int nlh_search_run(const void* handle, const Parameters* parameters, float* output,
                   double* seconds, std::uint64_t* groups) {
    try {
        if (!handle || !parameters || !output || !seconds || !groups)
            throw std::invalid_argument("invalid native search call");
        const auto start = Clock::now();
        const auto& source = *static_cast<const Source*>(handle);
        const auto& p = *parameters;
        nss::NlhImageOptions options;
        for (int s = 0; s < 2; ++s) {
            options.block[s] = p.block[s]; options.step[s] = p.step[s];
            options.group[s] = p.group[s]; options.q[s] = p.q[s];
            options.window[s] = p.window[s];
        }
        options.basic_iterations = p.basic_iterations;
        options.wiener_iterations = p.wiener_iterations;
        options.real_noise = p.real_noise != 0;
        options.basic_mix = p.mix; options.hard_strength = p.hard_strength;
        options.wiener_sigma_scale = p.wiener_scale;
        bool bypass = true;
        for (int c = 0; c < source.planes; ++c) bypass = bypass && source.input[0].sigma[c] == 0;
        *groups = 0;
        if (bypass) {
            std::copy(source.original.begin(), source.original.end(), output);
        } else {
            auto result = nss::nlh_image(source.input, nullptr, source.planes, 0, options);
            *groups = result.stats.groups;
            if (source.planes == 3) {
                if (result.denominator[0].planes[0].pixels != result.denominator[0].planes[1].pixels ||
                    result.denominator[0].planes[0].pixels != result.denominator[0].planes[2].pixels)
                    throw std::logic_error("inconsistent shared color aggregation");
                nss::nlh_yuv_to_rgb(result.numerator[0]);
            }
            const std::size_t count = std::size_t(source.width) * source.height;
            for (int c = 0; c < source.planes; ++c) {
                const auto& num = result.numerator[0].planes[c].pixels;
                const auto& den = result.denominator[0].planes[c].pixels;
                for (std::size_t i = 0; i < count; ++i) {
                    const double value = den[i] > 0 ? double(num[i]) / den[i] : source.original[c * count + i];
                    if (!std::isfinite(value)) throw std::runtime_error("nonfinite native search output");
                    output[c * count + i] = float(value);
                }
            }
        }
        *seconds = std::chrono::duration<double>(Clock::now() - start).count();
        return 0;
    } catch (const std::exception& error) { last_error = error.what(); return -1; }
}
}
