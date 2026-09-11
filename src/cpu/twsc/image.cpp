#include "nss/cpu_image.hpp"
#include "cpu/common/image_internal.hpp"
#include <array>

namespace nss {
ImageContributions twsc_image(const ImageSequence& input, const ImageSequence* reference,
                              int planes, int center, const TwscImageOptions& o) {
    using namespace image_detail;
    validate(input, reference, planes, center);
    if (o.lambda2 < 0 || !std::isfinite(o.lambda2) || o.delta < 0 || o.delta > 1 || !std::isfinite(o.delta))
        throw std::invalid_argument("nss.TWSC: invalid lambda2/delta");
    double sigma2 = 0;
    int active = 0;
    for (int c = 0; c < planes; ++c) if (input[center].sigma[c] > 0) { const double s = image_sigma_units(input[center], c); sigma2 += s * s; ++active; }
    const double sigma = active ? std::sqrt(sigma2 / active) : 0;
    const int block = o.block ? o.block : sigma <= 20 ? 7 : sigma <= 60 ? 8 : 9;
    const int group = o.group ? o.group : sigma <= 20 ? 70 : sigma <= 40 ? 90 : sigma <= 60 ? 120 : 140;
    const int iterations = o.iterations ? o.iterations : sigma <= 20 ? 8 : sigma <= 60 ? 12 : 14;
    if (block < 1 || block > 16 || group < 1 || group > 256 || iterations < 1 || iterations > 64 || o.step < 1 || o.step > block)
        throw std::invalid_argument("nss.TWSC: invalid resolved block/group/step/iters");
    ImageSearch search{block, o.step, group, o.window, o.radius, o.ps_num, o.ps_range};
    ImageSequence estimate = input;
    struct Job {
        TwscWorkspace work;
        ResourceVector<float> patches, column_sigma, column_weight;
        std::array<Match, 256> matches;
        int count = 0;
        TwscSolverStats stats;
    };
    ResourceVector<float> row_sigma;
    ImageFilterStats stats;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (iteration > 0 && o.delta != 0) for (std::size_t t = 0; t < input.size(); ++t) for (int c = 0; c < planes; ++c) {
            auto& e = estimate[t].planes[c].pixels; const auto& y = input[t].planes[c].pixels;
            for (std::size_t i = 0; i < e.size(); ++i) e[i] = finite_float(double(e[i]) + o.delta * (double(y[i]) - e[i]));
        }
        auto accum = accumulator(input, planes);
        std::array<bool, 3> used{};
        for (int first = 0; first < planes; ++first) {
            if (used[first] || input[center].sigma[first] == 0) continue;
            const int width = input[center].planes[first].width, height = input[center].planes[first].height;
            std::array<int, 3> channels{};
            int nch = 0;
            for (int c = first; c < planes; ++c) if (!used[c] && input[center].sigma[c] > 0 &&
                input[center].planes[c].width == width && input[center].planes[c].height == height) {
                channels[nch++] = c; used[c] = true;
            }
            const int area = block * block, m = area * nch;
            row_sigma.resize(m);
            const std::size_t estimate_bytes = checked_sum(checked_product({std::size_t(m), std::size_t(group), 160}), checked_product({std::size_t(group), std::size_t(group), 96}));
            std::size_t available = std::numeric_limits<std::size_t>::max();
            if (const auto budget = current_budget()) { const auto state = budget->snapshot(); available = state.limit - state.owned; }
            const int capacity = int(std::min<std::size_t>(16, std::max<std::size_t>(1, available / estimate_bytes)));
            ResourceVector<Job> jobs(capacity);
            for (auto& job : jobs) { job.patches.resize(std::size_t(m) * group); job.column_sigma.resize(group); job.column_weight.resize(group); }
            ResourceVector<const float*> guides(input.size() * nch);
            const auto& guide_images = reference ? *reference : estimate;
            for (std::size_t t = 0; t < input.size(); ++t) for (int c = 0; c < nch; ++c)
                guides[t * nch + c] = guide_images[t].planes[channels[c]].pixels.data();
            const int start = iteration + 1 == iterations ? center : 0;
            const int end = iteration + 1 == iterations ? center + 1 : int(input.size());
            for (int t0 = start; t0 < end; ++t0) {
                for (int c = 0; c < nch; ++c) std::fill_n(row_sigma.data() + c * area, area, input[t0].sigma[channels[c]]);
                int pending = 0;
                auto flush = [&]() {
                    std::array<TwscFullBatchItem, 16> items;
                    for (int i = 0; i < pending; ++i) {
                        auto& job = jobs[i];
                        items[i] = {job.patches.data(), m, job.count, m, row_sigma.data(), job.column_sigma.data(), job.column_weight.data(), &o.solver, &job.work, &job.stats};
                    }
                    twsc_filter_full_batch(items.data(), pending);
                    // Commit in query order, independent of shape buckets or SIMD lanes.
                    for (int i = 0; i < pending; ++i) {
                        const auto& job = jobs[i]; const auto& result = job.stats;
                        ++stats.groups; stats.max_iteration_groups += !result.converged; stats.double_svd_groups += result.double_svd;
                        stats.max_sylvester_residual = std::max(stats.max_sylvester_residual, result.sylvester_residual);
                        for (int j = 0; j < job.count; ++j) for (int c = 0; c < nch; ++c) {
                            const auto& match = job.matches[j];
                            add_patch(accum[match.t].planes[channels[c]], match.x, match.y, block, job.patches.data() + j * m + c * area, job.column_weight[j]);
                        }
                    }
                    pending = 0;
                };
                raster(width, height, block, o.step, [&](int x, int y) {
                    auto& job = jobs[pending];
                    const int n = image_match(guides.data(), int(input.size()), nch, width, height, t0, x, y, search, job.matches.data());
                    job.count = n;
                    for (int j = 0; j < n; ++j) {
                        const auto& match = job.matches[j];
                        double rms2 = 0, residual = 0;
                        for (int c = 0; c < nch; ++c) {
                            const int channel = channels[c];
                            const auto& e = estimate[match.t].planes[channel]; const auto& original = input[match.t].planes[channel];
                            float* patch = job.patches.data() + j * m + c * area;
                            pack_patch(patch, area, e.pixels.data(), width, match.x, match.y, block, width, height);
                            const double noise = input[match.t].sigma[channel]; rms2 += noise * noise;
                            if (iteration > 0) for (int iy = 0; iy < block; ++iy) for (int ix = 0; ix < block; ++ix) {
                                const double d = double(original.pixels[(match.y + iy) * width + match.x + ix]) - patch[iy * block + ix]; residual += d * d;
                            }
                        }
                        const double col_noise = o.lambda2 * std::sqrt(std::abs(rms2 / nch - residual / m));
                        job.column_sigma[j] = finite_float(col_noise);
                    }
                    if (++pending == capacity) flush();
                });
                if (pending) flush();
            }
        }
        if (iteration + 1 == iterations) return contributions(accum, input, planes, stats);
        estimate = finish(accum, input, planes);
    }
    throw std::logic_error("nss.TWSC: no iteration executed");
}
} // namespace nss
