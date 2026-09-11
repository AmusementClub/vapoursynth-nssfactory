#pragma once
#include "nss/cpu_image.hpp"
#include "nss/cpu_api.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace nss::image_detail {
struct AccumPlane { ResourceVector<double> numerator, denominator; int width = 0, height = 0; };
struct AccumFrame { std::array<AccumPlane, 3> planes; };
using Accumulator = ResourceVector<AccumFrame>;

inline Accumulator accumulator(const ImageSequence& input, int planes) {
    Accumulator a(input.size());
    for (std::size_t t = 0; t < input.size(); ++t) for (int c = 0; c < planes; ++c) {
        const auto& p = input[t].planes[c]; auto& out = a[t].planes[c];
        out.width = p.width; out.height = p.height;
        out.numerator.assign(p.pixels.size(), 0); out.denominator.assign(p.pixels.size(), 0);
    }
    return a;
}
inline float finite_float(double x) {
    if (!std::isfinite(x) || std::abs(x) > std::numeric_limits<float>::max()) throw std::runtime_error("nss: unrepresentable image estimate");
    return float(x);
}
inline ImageSequence finish(const Accumulator& a, const ImageSequence& fallback, int planes) {
    ImageSequence result(fallback.size());
    for (std::size_t t = 0; t < a.size(); ++t) {
        result[t].sigma = fallback[t].sigma;
        result[t].sigma_units = fallback[t].sigma_units;
        for (int c = 0; c < planes; ++c) {
            const auto& p = a[t].planes[c];
            auto& out = result[t].planes[c]; out = ImagePlane(p.width, p.height);
            for (std::size_t i = 0; i < out.pixels.size(); ++i)
                out.pixels[i] = p.denominator[i] > 0 ? finite_float(p.numerator[i] / p.denominator[i]) : fallback[t].planes[c].pixels[i];
        }
    }
    return result;
}
inline ImageContributions contributions(const Accumulator& a, const ImageSequence& input, int planes, ImageFilterStats stats) {
    ImageContributions result; result.stats = stats;
    result.numerator.resize(a.size()); result.denominator.resize(a.size());
    for (std::size_t t = 0; t < a.size(); ++t) {
        result.numerator[t].sigma = input[t].sigma;
        for (int c = 0; c < planes; ++c) {
            const auto& p = a[t].planes[c];
            auto& num = result.numerator[t].planes[c]; auto& den = result.denominator[t].planes[c];
            num = ImagePlane(p.width, p.height); den = ImagePlane(p.width, p.height);
            for (std::size_t i = 0; i < num.pixels.size(); ++i) { num.pixels[i] = finite_float(p.numerator[i]); den.pixels[i] = finite_float(p.denominator[i]); }
        }
    }
    return result;
}
inline void add_patch(AccumPlane& a, int x, int y, int block, const float* patch, double weight) {
    for (int iy = 0; iy < block; ++iy) for (int ix = 0; ix < block; ++ix) {
        const std::size_t index = std::size_t(y + iy) * a.width + x + ix;
        a.numerator[index] += double(patch[iy * block + ix]) * weight; a.denominator[index] += weight;
    }
}
inline void add_pixel_matrix(AccumPlane& a, int x, int y, int block, const double* num, const double* den) {
    for (int iy = 0; iy < block; ++iy) for (int ix = 0; ix < block; ++ix) {
        const std::size_t index = std::size_t(y + iy) * a.width + x + ix;
        a.numerator[index] += num[iy * block + ix]; a.denominator[index] += den[iy * block + ix];
    }
}
template<class F> inline void raster(int width, int height, int block, int step, F&& f) {
    if (width < block || height < block || step < 1 || step > block) throw std::invalid_argument("nss: selected plane is smaller than block_size or invalid step");
    // Include the last valid row/column once, with stable query order.
    for (int y = 0;; y = std::min(y + step, height - block)) {
        for (int x = 0;; x = std::min(x + step, width - block)) { f(x, y); if (x == width - block) break; }
        if (y == height - block) break;
    }
}
inline void validate(const ImageSequence& input, const ImageSequence* reference, int planes, int center) {
    if (input.empty() || input.size() > 33 || center < 0 || center >= int(input.size()) || planes < 1 || planes > 3 ||
        (reference && reference->size() != input.size())) throw std::invalid_argument("nss: invalid image sequence");
    for (std::size_t t = 0; t < input.size(); ++t) for (int c = 0; c < planes; ++c) {
        const auto& p = input[t].planes[c];
        if (p.width < 1 || p.height < 1 || p.width != input[0].planes[c].width || p.height != input[0].planes[c].height ||
            p.pixels.size() != std::size_t(p.width) * p.height || !(input[t].sigma[c] >= 0) || !std::isfinite(input[t].sigma[c]))
            throw std::invalid_argument("nss: invalid image plane/noise");
        for (float x : p.pixels) if (!std::isfinite(x)) throw std::invalid_argument("nss: nonfinite input sample");
        if (reference) {
            const auto& rp = (*reference)[t].planes[c];
            if (rp.width != p.width || rp.height != p.height || rp.pixels.size() != p.pixels.size()) throw std::invalid_argument("nss: invalid matching reference shape");
            for (float x : rp.pixels) if (!std::isfinite(x)) throw std::invalid_argument("nss: nonfinite matching reference");
        }
    }
}
} // namespace nss::image_detail
