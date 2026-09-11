#include "nss/cpu_image.hpp"
#include "nss/cpu_nlh.hpp"
#include "nss/cpu_nlh_full.hpp"
#include "cpu/common/image_internal.hpp"
#include "cpu/nlh/defaults.hpp"
#include <array>

namespace nss {
namespace {
using namespace image_detail;
Accumulator nlh_pass(const ImageSequence& data, const ImageSequence& basic, const ImageSequence* external,
                     int planes, int center, int block, int step, int group, int q, int window,
                     const NlhImageOptions& options, bool wiener, ImageFilterStats& stats) {
    auto accum = accumulator(data, planes);
    ImageSearch search{block, step, group, window, options.radius, options.ps_num, options.ps_range};
    std::array<bool, 3> used{};
    std::array<NlhWorkspace, 3> work;
    const std::size_t storage = std::size_t(block * block) * group;
    ResourceVector<float> guide_group(storage), pixels(storage * planes), reference(storage * planes);
    ResourceVector<int> indices(std::size_t(block * block) * q);
    std::array<Match, 64> matches;
    const auto& matching = external ? *external : wiener ? basic : data;
    for (int first = 0; first < planes; ++first) {
        if (used[first]) continue;
        const int width = data[center].planes[first].width, height = data[center].planes[first].height;
        std::array<int, 3> channels{};
        int nch = 0;
        for (int c = first; c < planes; ++c) if (!used[c] && data[center].planes[c].width == width && data[center].planes[c].height == height) {
            channels[nch++] = c; used[c] = true;
        }
        bool all_off = true;
        for (const auto& frame : data) for (int c = 0; c < nch; ++c) all_off = all_off && frame.sigma[channels[c]] == 0;
        if (all_off) continue;
        ResourceVector<ImagePlane> resized(data.size());
        ResourceVector<const float*> guides(data.size());
        for (std::size_t t = 0; t < data.size(); ++t) {
            const auto& luma = matching[t].planes[0];
            if (luma.width == width && luma.height == height) guides[t] = luma.pixels.data();
            else { resized[t] = image_area_guide(luma, width, height); guides[t] = resized[t].pixels.data(); }
        }
        const int m = block * block;
        const int start = wiener ? center : 0, end = wiener ? center + 1 : int(data.size());
        for (int t0 = start; t0 < end; ++t0) {
            raster(width, height, block, step, [&](int x, int y) {
                const int found = image_match(guides.data(), int(data.size()), 1, width, height, t0, x, y, search, matches.data());
                int n = 1;
                while (n * 2 <= found) n *= 2;
                for (int j = 0; j < n; ++j) {
                    const auto& mm = matches[j];
                    pack_patch(guide_group.data() + j * m, m, guides[mm.t], width, mm.x, mm.y, block, width, height);
                }
                pixel_match(guide_group.data(), m, n, m, q, indices.data());
                std::array<NlhGroupOptions, 3> configs;
                std::array<NlhFullBatchItem, 3> batch;
                for (int c = 0; c < nch; ++c) {
                    const int channel = channels[c];
                    for (int j = 0; j < n; ++j) {
                        const auto& mm = matches[j];
                        pack_patch(pixels.data() + c * storage + j * m, m, data[mm.t].planes[channel].pixels.data(), width, mm.x, mm.y, block, width, height);
                        if (wiener) pack_patch(reference.data() + c * storage + j * m, m, basic[mm.t].planes[channel].pixels.data(), width, mm.x, mm.y, block, width, height);
                    }
                    configs[c] = {q, data[t0].sigma[channel], wiener, options.hard_strength, options.wiener_iterations, options.wiener_sigma_scale};
                    batch[c] = {pixels.data() + c * storage, wiener ? reference.data() + c * storage : nullptr, m, n, m, &configs[c], indices.data(), &work[c]};
                }
                nlh_filter_full_batch(batch.data(), nch);
                for (int c = 0; c < nch; ++c) {
                    const int channel = channels[c];
                    ++stats.groups;
                    for (int j = 0; j < n; ++j) {
                        const auto& mm = matches[j];
                        add_pixel_matrix(accum[mm.t].planes[channel], mm.x, mm.y, block, work[c].numerator.data() + j * m, work[c].denominator.data() + j * m);
                    }
                }
            });
        }
    }
    return accum;
}
} // namespace

void nlh_estimate_frame_sigma(ImageFrame& frame, const ImageFrame& guide, int planes) {
    if (planes < 1 || planes > 3) throw std::invalid_argument("nss.NLH: invalid noise-estimation plane count");
    std::array<bool, 3> used{};
    for (int first = 0; first < planes; ++first) {
        if (used[first]) continue;
        const int width = frame.planes[first].width, height = frame.planes[first].height;
        if (width < 8 || height < 8)
            throw std::invalid_argument("nss: blind noise estimation requires an 8x8 or larger plane");
        std::array<int, 3> channels{};
        int count = 0;
        for (int c = first; c < planes; ++c) {
            if (!used[c] && frame.planes[c].width == width && frame.planes[c].height == height) {
                channels[count++] = c;
                used[c] = true;
            }
        }
        const auto& luma = guide.planes[0];
        ImagePlane resized;
        const ImagePlane* matching = &luma;
        if (luma.width != width || luma.height != height) {
            resized = image_area_guide(luma, width, height);
            matching = &resized;
        }
        if (count == 1) {
            frame.sigma[first] = nlh_estimate_sigma(frame.planes[first], *matching);
            frame.sigma_units[first] = double(frame.sigma[first]) * 255;
            continue;
        }
        constexpr int m = 64, q = 4;
        ImageSearch search;
        std::array<Match, 16> matches;
        std::array<float, m * 16> guide_group{};
        std::array<std::array<float, m * 16>, 3> pixels{};
        std::array<int, m * q> indices{};
        std::array<double, 3> totals{};
        std::uint64_t groups = 0;
        const float* guides[]{matching->pixels.data()};
        for (int y = 0; y <= height - 8; ++y) for (int x = 0; x <= width - 8; ++x) {
            const int n = image_match(guides, 1, 1, width, height, 0, x, y, search, matches.data());
            for (int j = 0; j < n; ++j) {
                const auto& mm = matches[j];
                pack_patch(guide_group.data() + j * m, m, guides[0], width, mm.x, mm.y, 8, width, height);
                for (int c = 0; c < count; ++c)
                    pack_patch(pixels[c].data() + j * m, m, frame.planes[channels[c]].pixels.data(),
                               width, mm.x, mm.y, 8, width, height);
            }
            pixel_match(guide_group.data(), m, n, m, q, indices.data());
            for (int c = 0; c < count; ++c) {
                double local = 0;
                for (int row = 0; row < m; ++row) for (int k = 1; k < q; ++k) {
                    const int neighbor = indices[row * q + k];
                    double d2 = 0;
                    for (int j = 0; j < n; ++j) {
                        const double d = double(pixels[c][row + j * m]) - pixels[c][neighbor + j * m];
                        d2 += d * d;
                    }
                    local += std::sqrt(d2 / n);
                }
                totals[c] += local / (m * (q - 1));
            }
            ++groups;
        }
        for (int c = 0; c < count; ++c) {
            const double sigma = totals[c] / double(groups);
            if (!std::isfinite(sigma)) throw std::runtime_error("nss: nonfinite blind noise estimate");
            frame.sigma[channels[c]] = float(sigma);
            frame.sigma_units[channels[c]] = double(frame.sigma[channels[c]]) * 255;
        }
    }
}

NlhImageOptions nlh_resolve_options(std::span<const ImageFrame> input, int planes, int center,
                                   const NlhImageOptions& requested) {
    if (input.empty() || center < 0 || center >= int(input.size()) || planes < 1 || planes > 3)
        throw std::invalid_argument("nss.NLH: invalid preset context");
    double sigma = 0;
    int available = 16;
    for (int c = 0; c < planes; ++c) {
        const double units = image_sigma_units(input[center], c);
        if (!(units >= 0) || !std::isfinite(units)) throw std::invalid_argument("nss.NLH: invalid preset sigma");
        sigma = std::max(sigma, units);
        for (const auto& frame : input) if (frame.sigma[c] > 0)
            available = std::min({available, frame.planes[c].width, frame.planes[c].height});
    }
    if (available < 2) throw std::invalid_argument("nss: selected plane is smaller than block_size");
    const auto& preset = requested.real_noise ? nlh_detail::kReal :
                         sigma <= 50 ? nlh_detail::kAwgnLow : nlh_detail::kAwgnHigh;
    auto o = requested;
    for (int s = 0; s < 2; ++s) {
        o.block[s] = o.block[s] ? o.block[s] : std::min(preset.block[s], available);
        if (o.block[s] < 2 || o.block[s] > 16 || o.block[s] > available)
            throw std::invalid_argument("nss.NLH: invalid resolved block_size");
        o.step[s] = o.step[s] ? o.step[s] : std::min(preset.step[s], o.block[s]);
        if (!o.q[s]) {
            o.q[s] = preset.q[s];
            while (o.q[s] > o.block[s] * o.block[s]) o.q[s] /= 2;
        }
        o.group[s] = o.group[s] ? o.group[s] : preset.group[s];
        o.window[s] = o.window[s] ? o.window[s] : preset.window[s];
        if (o.q[s] < 2 || o.q[s] > 16 || (o.q[s] & (o.q[s] - 1)) ||
            o.q[s] > o.block[s] * o.block[s] || o.group[s] < 2 || o.group[s] > 64 ||
            (o.group[s] & (o.group[s] - 1)) || o.step[s] < 1 || o.step[s] > o.block[s] ||
            o.window[s] < 1 || o.window[s] > 129)
            throw std::invalid_argument("nss.NLH: invalid resolved stage shape");
    }
    o.basic_iterations = o.basic_iterations ? o.basic_iterations : preset.basic_iterations;
    o.wiener_iterations = o.wiener_iterations ? o.wiener_iterations : preset.wiener_iterations;
    if (o.basic_mix == -1) o.basic_mix = preset.basic_mix;
    if (o.hard_strength == -1) o.hard_strength = preset.hard_strength;
    if (o.wiener_sigma_scale == -1) o.wiener_sigma_scale = preset.wiener_sigma_scale;
    if (o.basic_iterations < 1 || o.basic_iterations > 64 || o.wiener_iterations < 1 || o.wiener_iterations > 64 ||
        !(o.basic_mix >= 0 && o.basic_mix <= 1) || !std::isfinite(o.basic_mix) ||
        !(o.hard_strength >= 0) || !std::isfinite(o.hard_strength) ||
        !(o.wiener_sigma_scale >= 0) || !std::isfinite(o.wiener_sigma_scale) ||
        o.radius < 0 || o.radius > 16 || o.ps_num < 1 || o.ps_num > std::min(o.group[0], o.group[1]) ||
        o.ps_range < 1 || o.ps_range > 64)
        throw std::invalid_argument("nss.NLH: invalid resolved iteration, coefficient or temporal parameters");
    return o;
}

ImageContributions nlh_image(const ImageSequence& input, const ImageSequence* reference,
                             int planes, int center, const NlhImageOptions& requested) {
    using namespace image_detail;
    validate(input, reference, planes, center);
    const auto o = nlh_resolve_options(input, planes, center, requested);
    ImageSequence basic = input;
    ImageFilterStats stats;
    for (int iteration = 0; iteration < o.basic_iterations; ++iteration) {
        for (std::size_t t = 0; t < input.size(); ++t) for (int c = 0; c < planes; ++c) {
            auto& e = basic[t].planes[c].pixels; const auto& y = input[t].planes[c].pixels;
            for (std::size_t i = 0; i < e.size(); ++i) e[i] = finite_float(o.basic_mix * e[i] + (1 - o.basic_mix) * y[i]);
        }
        const auto accum = nlh_pass(basic, basic, reference, planes, center, o.block[0], o.step[0], o.group[0], o.q[0], o.window[0], o, false, stats);
        basic = finish(accum, input, planes);
    }
    const auto final = nlh_pass(input, basic, reference, planes, center, o.block[1], o.step[1], o.group[1], o.q[1], o.window[1], o, true, stats);
    return contributions(final, input, planes, stats);
}
} // namespace nss
