#include "nss/cpu_image.hpp"
#include "nss/cpu_api.hpp"
#include "nss/cpu_nlh.hpp"
#include "cpu/bm/matcher.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace nss {
namespace {
#if NSS_ALIGNMENT_GENERIC
using ImageTopK = detail::StableTopK;
#else
using ImageTopK = detail::SortedTopK;
#endif
float image_ssd(const float* a, const float* b, int stride, int block) {
#if NSS_ALIGNMENT_GENERIC
    float value = 0;
    for (int y = 0; y < block; ++y) for (int x = 0; x < block; ++x) {
        const float d = a[y * stride + x] - b[y * stride + x]; value += d * d;
    }
    return value;
#else
    return ssd_block(a, stride, b, stride, block);
#endif
}
}

int image_match(const float* const* guides, int frames, int nch, int width, int height,
                int t0, int x0, int y0, const ImageSearch& cfg, Match* matches) {
    if (!guides || !matches || frames < 1 || frames > 33 || nch < 1 || nch > 3 || t0 < 0 || t0 >= frames ||
        cfg.block < 1 || cfg.block > 16 || width < cfg.block || height < cfg.block ||
        x0 < 0 || x0 > width - cfg.block || y0 < 0 || y0 > height - cfg.block ||
        cfg.group < 1 || cfg.group > 256 || cfg.window < 1 || cfg.window > 129 || cfg.radius < 0 || cfg.radius > 16 ||
        cfg.ps_num < 1 || cfg.ps_num > cfg.group || cfg.ps_range < 1 || cfg.ps_range > 64)
        throw std::invalid_argument("nss: invalid full-algorithm search configuration");
    auto distance = [&](int t, int x, int y) {
        float sum = 0;
        for (int c = 0; c < nch; ++c)
            sum += image_ssd(guides[t0 * nch + c] + y0 * width + x0,
                             guides[t * nch + c] + y * width + x, width, cfg.block);
        if (!std::isfinite(sum)) throw std::runtime_error("nss: unrepresentable patch distance");
        return sum;
    };
    matches[0] = Match{x0, y0, t0, 0, 0};
    if (cfg.group == 1) return 1;
    ImageTopK spatial(matches + 1, cfg.group - 1);
    const int lo = cfg.window / 2, hi = cfg.window - lo - 1;
    for (int y = std::max(0, y0 - lo); y <= std::min(height - cfg.block, y0 + hi); ++y)
        for (int x = std::max(0, x0 - lo); x <= std::min(width - cfg.block, x0 + hi); ++x) {
            if (x == x0 && y == y0) continue;
            spatial.add(Match{x, y, t0, distance(t0, x, y), std::uint32_t(1 + y * width + x)});
        }
    const int n = 1 + spatial.finish();
    if (cfg.radius == 0 || frames == 1) return n;
    std::array<Match, 256> seeds{}, centers{}, local{};
    const int seed_count = std::min(n, cfg.ps_num);
    std::copy_n(matches, seed_count, seeds.data());
    // StableTopK/SortedTopK lack a shared adopt interface: insert the spatial
    // prefix into separate storage, preserving its exact retained set.
    std::array<Match, 256> retained{};
    ImageTopK global(retained.data(), cfg.group - 1);
    for (int i = 1; i < n; ++i) global.add(matches[i]);
    for (int direction : {-1, 1}) {
        std::copy_n(seeds.data(), seed_count, centers.data());
        int nc = seed_count;
        for (int delta = 1; delta <= cfg.radius; ++delta) {
            const int t = t0 + direction * delta;
            if (t < 0 || t >= frames) break;
            ImageTopK next(local.data(), cfg.ps_num);
            int ymin = height, ymax = -1;
            for (int i = 0; i < nc; ++i) {
                ymin = std::min(ymin, std::max(0, centers[i].y - cfg.ps_range));
                ymax = std::max(ymax, std::min(height - cfg.block, centers[i].y + cfg.ps_range));
            }
            for (int y = ymin; y <= ymax; ++y) {
                std::array<std::pair<int, int>, 256> intervals;
                int ni = 0;
                for (int i = 0; i < nc; ++i) if (std::abs(y - centers[i].y) <= cfg.ps_range)
                    intervals[ni++] = {std::max(0, centers[i].x - cfg.ps_range), std::min(width - cfg.block, centers[i].x + cfg.ps_range)};
                std::sort(intervals.begin(), intervals.begin() + ni);
                int done = -1;
                for (int i = 0; i < ni; ++i) {
                    for (int x = std::max(done + 1, intervals[i].first); x <= intervals[i].second; ++x)
                        next.add(Match{x, y, t, distance(t, x, y), std::uint32_t(1 + y * width + x)});
                    done = std::max(done, intervals[i].second);
                }
            }
            nc = next.finish();
            for (int i = 0; i < nc; ++i) global.add(local[i]);
            std::copy_n(local.data(), nc, centers.data());
            if (nc == 0) break;
        }
    }
    const int count = global.finish();
    std::copy_n(retained.data(), count, matches + 1);
    return 1 + count;
}

float nlh_estimate_sigma(const ImagePlane& channel, const ImagePlane& guide) {
    if (channel.width != guide.width || channel.height != guide.height || channel.width < 8 || channel.height < 8)
        throw std::invalid_argument("nss: blind noise estimation requires an 8x8 or larger plane");
    ImageSearch cfg;
    const int m = 64, q = 4;
    std::array<Match, 16> matches;
    std::array<float, 64 * 16> g{}, pixels{};
    std::array<int, 64 * 4> indices{};
    const float* guides[] = {guide.pixels.data()};
    double total = 0;
    std::uint64_t groups = 0;
    for (int y = 0; y <= channel.height - 8; ++y) for (int x = 0; x <= channel.width - 8; ++x) {
        const int n = image_match(guides, 1, 1, channel.width, channel.height, 0, x, y, cfg, matches.data());
        for (int j = 0; j < n; ++j) {
            const auto& mm = matches[j];
            pack_patch(g.data() + j * m, m, guide.pixels.data(), guide.width, mm.x, mm.y, 8, guide.width, guide.height);
            pack_patch(pixels.data() + j * m, m, channel.pixels.data(), channel.width, mm.x, mm.y, 8, channel.width, channel.height);
        }
        pixel_match(g.data(), m, n, m, q, indices.data());
        double local = 0;
        for (int row = 0; row < m; ++row) for (int k = 1; k < q; ++k) {
            const int neighbor = indices[row * q + k];
            double d2 = 0;
            for (int j = 0; j < n; ++j) {
                const double d = double(pixels[row + j * m]) - pixels[neighbor + j * m]; d2 += d * d;
            }
            local += std::sqrt(d2 / n);
        }
        total += local / (m * (q - 1)); ++groups;
    }
    const double result = total / double(groups);
    if (!std::isfinite(result)) throw std::runtime_error("nss: nonfinite blind noise estimate");
    return float(result);
}

void nlh_rgb_to_yuv(ImageFrame& frame, bool propagate_sigma) {
    constexpr double a[3][3]{{0.299, 0.587, 0.114}, {-0.168736607142857, -0.331263392857143, 0.5}, {0.5, -0.4186875, -0.0813125}};
    const std::size_t n = frame.planes[0].pixels.size();
    for (std::size_t i = 0; i < n; ++i) {
        const double r = frame.planes[0].pixels[i], g = frame.planes[1].pixels[i], b = frame.planes[2].pixels[i];
        for (int c = 0; c < 3; ++c) frame.planes[c].pixels[i] = float(a[c][0] * r + a[c][1] * g + a[c][2] * b);
    }
    if (propagate_sigma) {
        const auto sigma = frame.sigma;
        const auto original_units = frame.sigma_units;
        for (int c = 0; c < 3; ++c) {
            double variance = 0, units_variance = 0;
            for (int k = 0; k < 3; ++k) {
                variance += a[c][k] * a[c][k] * double(sigma[k]) * sigma[k];
                const double units = original_units[k] >= 0 ? original_units[k] : double(sigma[k]) * 255;
                units_variance += a[c][k] * a[c][k] * units * units;
            }
            frame.sigma[c] = float(std::sqrt(variance));
            frame.sigma_units[c] = std::sqrt(units_variance);
        }
    }
}
void nlh_yuv_to_rgb(ImageFrame& frame) {
    const std::size_t n = frame.planes[0].pixels.size();
    for (std::size_t i = 0; i < n; ++i) {
        const double y = frame.planes[0].pixels[i], u = frame.planes[1].pixels[i], v = frame.planes[2].pixels[i];
        frame.planes[0].pixels[i] = float(y + 1.402 * v);
        frame.planes[1].pixels[i] = float(y - (0.114 * 1.772 / 0.587) * u - (0.299 * 1.402 / 0.587) * v);
        frame.planes[2].pixels[i] = float(y + 1.772 * u);
    }
}
ImagePlane image_area_guide(const ImagePlane& luma, int width, int height) {
    if (width < 1 || height < 1 || luma.width % width || luma.height % height)
        throw std::invalid_argument("nss: unsupported luminance guide grid");
    ImagePlane guide(width, height);
    const int sx = luma.width / width, sy = luma.height / height;
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        double sum = 0;
        for (int dy = 0; dy < sy; ++dy) for (int dx = 0; dx < sx; ++dx) sum += luma.pixels[(y * sy + dy) * luma.width + x * sx + dx];
        guide.pixels[y * width + x] = float(sum / (sx * sy));
    }
    return guide;
}
} // namespace nss
