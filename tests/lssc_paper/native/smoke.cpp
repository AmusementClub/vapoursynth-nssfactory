// Standalone sanitizer/ABI smoke. Numerical SVD/KKT oracles live in test_native.py.
#include "explore.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

namespace {
using nss::explore::Options;
using nss::explore::Stats;
bool pursuit(int mode) {
    constexpr int m = 81, k = 512, g = 129;
    std::vector<double> d(m * k, 0), y(m * g, 0), output(m * g), coefficients(k * g), history(m + 1);
    std::vector<int> support(m);
    for (int j = 0; j < k; ++j) d[j % m + j * m] = 1;
    for (int j = 0; j < g; ++j) {
        y[3 + j * m] = std::sin(double(j + 1));
        y[17 + j * m] = .4 * std::cos(double(j + 1));
    }
    Options options;
    options.precision = 1; options.correlation = mode;
    Stats stats;
    if (nss_explore_solve(d.data(), y.data(), m, k, g, 1e-12, &options,
                          output.data(), coefficients.data(), support.data(), history.data(), &stats)) return false;
    if (stats.last_rank != 2 || stats.last_residual > 1e-12) return false;
    for (std::size_t i = 0; i < y.size(); ++i) if (std::abs(y[i] - output[i]) > 1e-12) return false;
    options.memory_limit_bytes = 1024;
    if (!nss_explore_solve(d.data(), y.data(), m, k, g, 1e-12, &options,
                           output.data(), coefficients.data(), support.data(), history.data(), &stats)) return false;
    return std::strstr(nss_explore_error(), "memory_limit_mb exceeded") != nullptr && stats.tracked_peak_bytes <= 1024;
}

bool image(int mode, bool learning) {
    constexpr int width = 32, height = 29, block = 9, m = 81, k = 512;
    std::vector<double> d(m * k, 0), pixels(width * height), output(width * height), pilot(width * height), learned(m * k);
    std::vector<std::uint32_t> coverage(width * height);
    std::vector<double> epsilon(1025, 0);
    for (int j = 0; j < k; ++j) d[j % m + j * m] = 1;
    for (std::size_t i = 0; i < pixels.size(); ++i) pixels[i] = .5 + .13 * std::sin(double(i * 17 + i / 7));
    // Explicit smoke budgets. The independent Python tests verify the actual
    // vectorized chi-square table against the frozen reference.
    for (std::size_t g = 1; g < epsilon.size(); ++g) epsilon[g] = .05 * .05 * m * g * 1.25;
    Options options;
    options.sigma = .05; options.precision = 1; options.correlation = mode; options.match_precision = 1;
    if (learning) {
        options.image_passes = 1; options.learning_samples = 128;
        options.learning_iterations = 24; options.learning_groups = 4;
    }
    Stats stats;
    if (nss_explore_denoise(pixels.data(), width, height, d.data(), block, k, epsilon.data(), int(epsilon.size()),
                            &options, output.data(), pilot.data(), learned.data(), coverage.data(), &stats)) return false;
    if (stats.pilot_patches != (width - block + 1) * (height - block + 1) || stats.capped_groups) return false;
    if (stats.dictionary_versions != (learning ? 2u : 1u)) return false;
    for (std::size_t i = 0; i < output.size(); ++i)
        if (!coverage[i] || !std::isfinite(output[i]) || !std::isfinite(pilot[i])) return false;
    return true;
}
}

int main() {
    if (nss_explore_abi() != 1 || nss_explore_options_size() != sizeof(Options) ||
        nss_explore_stats_size() != sizeof(Stats)) return 1;
    for (int mode : {1, 3, 7}) if (!pursuit(mode) || !image(mode, true)) {
        std::cerr << "native smoke failed: " << nss_explore_error() << '\n';
        return 2;
    }
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) threads.emplace_back([&, i] {
        if (!pursuit(i % 2 ? 7 : 3) || !image(i % 2 ? 7 : 3, false)) ++failures;
    });
    for (auto& thread : threads) thread.join();
    if (failures) return 3;
    std::cout << "native ABI, 512 atoms, odd geometry, learning, budget failure, concurrent calls: PASS\n";
    return 0;
}
