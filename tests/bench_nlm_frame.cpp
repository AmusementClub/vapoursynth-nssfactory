#include "nss/cpu_api.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#endif

#ifndef NSS_VERSION_STRING
#define NSS_VERSION_STRING "unknown"
#endif

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kD = 1;
constexpr int kA = 2;
constexpr int kS = 4;
constexpr float kH = 1.2f;
constexpr float kWref = 1.0f;
constexpr std::size_t kMaxScratch = 1u << 20;

struct Plane {
    int width = 0;
    int height = 0;
    int stride = 0;
    std::vector<float> pixels;
};

unsigned env_unsigned(const char* name, unsigned fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return fallback;
    }
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    return end != value && *end == '\0' && parsed <= 0xffffffffUL ? static_cast<unsigned>(parsed) : fallback;
}

Plane make_plane(int width, int height, unsigned seed) {
    Plane plane;
    plane.width = width;
    plane.height = height;
    plane.stride = (width + 15) & ~15;
    plane.pixels.assign(static_cast<std::size_t>(plane.stride) * static_cast<std::size_t>(height), 0.0f);
    std::uint32_t state = seed ? seed : 1u;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            state = state * 1664525u + 1013904223u;
            const float noise = static_cast<float>(state >> 8) * (1.0f / 16777216.0f);
            const float pattern = 0.03f * std::sin(static_cast<float>(x) / 31.0f) *
                                  std::cos(static_cast<float>(y) / 23.0f);
            plane.pixels[static_cast<std::size_t>(y * plane.stride + x)] = 0.35f + pattern + 0.2f * noise;
        }
    }
    return plane;
}

float h2_inv_norm() {
    const int diameter = 2 * kS + 1;
    return (255.0f * 255.0f) / (3.0f * kH * kH * static_cast<float>(diameter * diameter));
}

struct LegacyWorkspace {
    explicit LegacyWorkspace(int width, int height, int stride)
        : size(static_cast<std::size_t>(height) * static_cast<std::size_t>(stride)), weight(size), wdst(size),
          max_weight(size), temp(size), temp_bwd(size), temp_fwd(size), row(static_cast<std::size_t>(width)) {}

    std::size_t size;
    std::vector<float> weight;
    std::vector<float> wdst;
    std::vector<float> max_weight;
    std::vector<float> temp;
    std::vector<float> temp_bwd;
    std::vector<float> temp_fwd;
    std::vector<float> row;
};

void run_legacy(const std::array<Plane, 3>& frames, LegacyWorkspace& ws, std::vector<float>& output) {
    const int width = frames[1].width;
    const int height = frames[1].height;
    const int stride = frames[1].stride;
    std::fill(ws.weight.begin(), ws.weight.end(), 0.0f);
    std::fill(ws.wdst.begin(), ws.wdst.end(), 0.0f);
    std::fill(ws.max_weight.begin(), ws.max_weight.end(), std::numeric_limits<float>::epsilon());
    const float scale = h2_inv_norm();
    const int span = 2 * kA + 1;
    const float* center = frames[1].pixels.data();

    for (int i = -kD; i <= 0; ++i) {
        const int backward = kD + i;
        const int forward = kD - i;
        for (int oy = -kA; oy <= kA; ++oy) {
            for (int ox = -kA; ox <= kA; ++ox) {
                if (i * span * span + oy * span + ox >= 0) {
                    continue;
                }
                nss::nlm_distance_luma_f32(ws.temp_bwd.data(), center, frames[backward].pixels.data(), ox, oy,
                                           width, height, stride);
                nss::nlm_horizontal(ws.temp.data(), ws.temp_bwd.data(), kS, width, height, stride);
                nss::nlm_vertical_welsch(ws.temp_bwd.data(), ws.temp.data(), kS, scale, width, height, stride,
                                         ws.row.data());
                if (i == 0) {
                    nss::nlm_accum_ch1(ws.weight.data(), ws.wdst.data(), ws.max_weight.data(), center, center,
                                       ws.temp_bwd.data(), ws.temp_bwd.data(), ox, oy, width, height, stride);
                    continue;
                }
                nss::nlm_distance_luma_f32(ws.temp_fwd.data(), frames[forward].pixels.data(), center, ox, oy,
                                           width, height, stride);
                nss::nlm_horizontal(ws.temp.data(), ws.temp_fwd.data(), kS, width, height, stride);
                nss::nlm_vertical_welsch(ws.temp_fwd.data(), ws.temp.data(), kS, scale, width, height, stride,
                                         ws.row.data());
                nss::nlm_accum_ch1(ws.weight.data(), ws.wdst.data(), ws.max_weight.data(),
                                   frames[backward].pixels.data(), frames[forward].pixels.data(), ws.temp_bwd.data(),
                                   ws.temp_fwd.data(), ox, oy, width, height, stride);
            }
        }
    }
    nss::nlm_finish_ch1(output.data(), center, ws.weight.data(), ws.wdst.data(), ws.max_weight.data(), kWref, width,
                        height, stride);
}

#if defined(NSS_NLM_STRIPE_BENCH)

std::size_t scratch_budget() {
#if defined(_SC_LEVEL2_CACHE_SIZE)
    const long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
    if (l2 > 0) {
        return std::min(kMaxScratch, static_cast<std::size_t>(l2) / 2u);
    }
#endif
    return kMaxScratch;
}

struct StripeWorkspace {
    explicit StripeWorkspace(int width, int height, int stride) {
        const int halo = kA + kS;
        auto stripe_bytes = [&](int core) {
            const std::size_t core_rows = static_cast<std::size_t>(std::max(core, 0));
            const std::size_t ext_rows = std::min(static_cast<std::size_t>(height),
                                                  core_rows + 2u * static_cast<std::size_t>(halo));
            const std::size_t rows = 4u * core_rows + 2u * ext_rows;
            return rows * static_cast<std::size_t>(stride) * sizeof(float) +
                   static_cast<std::size_t>(width) * sizeof(float);
        };
        const std::size_t budget = std::min(kMaxScratch, scratch_budget());
        core_rows = 1;
        while (core_rows < height && stripe_bytes(core_rows + 1) <= budget) {
            ++core_rows;
        }
        const std::size_t ext_rows = std::min(static_cast<std::size_t>(height),
                                              static_cast<std::size_t>(core_rows + 2 * halo));
        const std::size_t floats = 4u * static_cast<std::size_t>(core_rows) * static_cast<std::size_t>(stride) +
                                   2u * ext_rows * static_cast<std::size_t>(stride) +
                                   static_cast<std::size_t>(width);
        storage.resize(floats);
    }

    int core_rows = 1;
    std::vector<float> storage;
};

void run_stripe(const std::array<Plane, 3>& frames, StripeWorkspace& ws, std::vector<float>& output) {
    const int width = frames[1].width;
    const int height = frames[1].height;
    const int stride = frames[1].stride;
    const int halo = kA + kS;
    const int span = 2 * kA + 1;
    const float scale = h2_inv_norm();

    for (int core0 = 0; core0 < height; core0 += ws.core_rows) {
        const int core1 = std::min(height, core0 + ws.core_rows);
        const int ext0 = halo >= core0 ? 0 : core0 - halo;
        const int ext1 = halo >= height - core1 ? height : core1 + halo;
        const int ext_height = ext1 - ext0;
        const int core_local = core0 - ext0;
        const int core_height = core1 - core0;
        const std::size_t core_size = static_cast<std::size_t>(core_height) * static_cast<std::size_t>(stride);
        const std::size_t ext_size = static_cast<std::size_t>(ext_height) * static_cast<std::size_t>(stride);

        float* weight = ws.storage.data();
        float* wdst = weight + core_size;
        float* max_weight = wdst + core_size;
        float* saved_bwd = max_weight + core_size;
        float* temp = saved_bwd + core_size;
        float* temp_bwd = temp + ext_size;
        float* row = temp_bwd + ext_size;
        std::fill(weight, weight + 2 * core_size, 0.0f);
        std::fill(max_weight, max_weight + core_size, std::numeric_limits<float>::epsilon());
        const float* center = frames[1].pixels.data() + static_cast<std::size_t>(ext0) * stride;

        for (int i = -kD; i <= 0; ++i) {
            const int backward = kD + i;
            const int forward = kD - i;
            const float* source_bwd = frames[backward].pixels.data() + static_cast<std::size_t>(ext0) * stride;
            const float* source_fwd = frames[forward].pixels.data() + static_cast<std::size_t>(ext0) * stride;
            for (int oy = -kA; oy <= kA; ++oy) {
                for (int ox = -kA; ox <= kA; ++ox) {
                    if (i * span * span + oy * span + ox >= 0) {
                        continue;
                    }
                    nss::nlm_distance_luma_f32(temp_bwd, center, source_bwd, ox, oy, width, ext_height, stride);
                    nss::nlm_horizontal(temp, temp_bwd, kS, width, ext_height, stride);
                    nss::nlm_vertical_welsch(temp_bwd, temp, kS, scale, width, ext_height, stride, row);
                    if (i == 0) {
                        nss::nlm_accum_ch1_range(weight, wdst, max_weight, source_bwd, source_bwd, temp_bwd, temp_bwd,
                                                 ox, oy, width, ext_height, stride, core_local,
                                                 core_local + core_height);
                        continue;
                    }
                    for (int y = 0; y < core_height; ++y) {
                        std::memcpy(saved_bwd + static_cast<std::size_t>(y) * stride,
                                    temp_bwd + static_cast<std::size_t>(core_local + y) * stride,
                                    static_cast<std::size_t>(stride) * sizeof(float));
                    }
                    nss::nlm_distance_luma_f32(temp_bwd, source_fwd, center, ox, oy, width, ext_height, stride);
                    nss::nlm_horizontal(temp, temp_bwd, kS, width, ext_height, stride);
                    const int temp2_base_y = std::clamp(core_local - oy, 0, ext_height - 1);
                    const int temp2_end_y = std::clamp(core_local + core_height - 1 - oy, 0, ext_height - 1) + 1;
                    nss::nlm_vertical_welsch_range(temp_bwd, temp, kS, scale, width, ext_height, stride, temp2_base_y,
                                                   temp2_end_y, row);
                    nss::nlm_accum_ch1_core_range(weight, wdst, max_weight, source_bwd, source_fwd, saved_bwd,
                                                  temp_bwd, ox, oy, width, ext_height, stride, core_local,
                                                  core_local + core_height, temp2_base_y);
                }
            }
        }
        nss::nlm_finish_ch1(output.data() + static_cast<std::size_t>(core0) * stride,
                            center + static_cast<std::size_t>(core_local) * stride, weight, wdst, max_weight, kWref,
                            width, core_height, stride);
    }
}

#endif

std::uint64_t output_hash(const std::vector<float>& output, int width, int height, int stride) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (int y = 0; y < height; ++y) {
        const auto* bytes = reinterpret_cast<const unsigned char*>(output.data() + static_cast<std::size_t>(y) * stride);
        for (std::size_t i = 0; i < static_cast<std::size_t>(width) * sizeof(float); ++i) {
            hash ^= bytes[i];
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

bool output_finite(const std::vector<float>& output, int width, int height, int stride) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (!std::isfinite(output[static_cast<std::size_t>(y) * stride + x])) {
                return false;
            }
        }
    }
    return true;
}

bool dump_output(const char* path, const std::vector<float>& output, int width, int height, int stride) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    for (int y = 0; file && y < height; ++y) {
        file.write(reinterpret_cast<const char*>(output.data() + static_cast<std::size_t>(y) * stride),
                   static_cast<std::streamsize>(static_cast<std::size_t>(width) * sizeof(float)));
    }
    return static_cast<bool>(file);
}

void usage(const char* argv0) {
    std::fprintf(stderr, "usage: %s legacy|stripe WIDTH HEIGHT ITERATIONS [--json] [--dump PATH]\n", argv0);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        usage(argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    const int width = std::atoi(argv[2]);
    const int height = std::atoi(argv[3]);
    const int iterations = std::atoi(argv[4]);
    bool json = false;
    const char* dump_path = nullptr;
    for (int i = 5; i < argc; ++i) {
        if (std::strcmp(argv[i], "--json") == 0) {
            json = true;
        } else if (std::strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            dump_path = argv[++i];
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (width <= 0 || height <= 0 || iterations <= 0 || (mode != "legacy" && mode != "stripe")) {
        usage(argv[0]);
        return 2;
    }
#if !defined(NSS_NLM_STRIPE_BENCH)
    if (mode == "stripe") {
        std::fprintf(stderr, "stripe mode is unavailable in this baseline build\n");
        return 2;
    }
#endif

    const unsigned seed = env_unsigned("NSS_BENCH_SEED", 42u);
    const unsigned frame = env_unsigned("NSS_BENCH_FRAME", 0u);
    const unsigned content_seed = seed + frame * 2654435761u;
    std::array<Plane, 3> frames{make_plane(width, height, content_seed), make_plane(width, height, content_seed + 1u),
                                make_plane(width, height, content_seed + 2u)};
    const int stride = frames[1].stride;
    std::vector<float> output(static_cast<std::size_t>(stride) * static_cast<std::size_t>(height));
    LegacyWorkspace legacy(width, height, stride);
#if defined(NSS_NLM_STRIPE_BENCH)
    StripeWorkspace stripe(width, height, stride);
#endif

    auto run_once = [&] {
        if (mode == "legacy") {
            run_legacy(frames, legacy, output);
        }
#if defined(NSS_NLM_STRIPE_BENCH)
        else {
            run_stripe(frames, stripe, output);
        }
#endif
    };
    if (std::getenv("NSS_NO_WARMUP") == nullptr) {
        run_once();
    }
    const auto start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        run_once();
    }
    const double wall_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    if (dump_path && !dump_output(dump_path, output, width, height, stride)) {
        std::fprintf(stderr, "failed to write output dump: %s\n", dump_path);
        return 1;
    }
    const bool finite = output_finite(output, width, height, stride);
    const std::uint64_t hash = output_hash(output, width, height, stride);
    const int core_rows =
#if defined(NSS_NLM_STRIPE_BENCH)
        mode == "stripe" ? stripe.core_rows : height;
#else
        height;
#endif
    if (json) {
        std::printf("{\"schema\":\"nssfactory.bench.nlm-frame.v1\",\"mode\":\"%s\","
                    "\"git_revision\":\"%s\",\"width\":%d,\"height\":%d,\"stride\":%d,"
                    "\"seed\":%u,\"frame_number\":%u,\"d\":%d,\"a\":%d,\"s\":%d,\"h\":%.9g,"
                    "\"wref\":%.9g,\"iterations\":%d,\"wall_time_ms\":%.17g,"
                    "\"milliseconds_per_iteration\":%.17g,\"core_rows\":%d,"
                    "\"output_hash\":\"%016llx\",\"output_finite\":%s}\n",
                    mode.c_str(), NSS_VERSION_STRING, width, height, stride, seed, frame, kD, kA, kS,
                    static_cast<double>(kH), static_cast<double>(kWref), iterations, wall_ms,
                    wall_ms / static_cast<double>(iterations), core_rows, static_cast<unsigned long long>(hash),
                    finite ? "true" : "false");
    } else {
        std::printf("nlm_frame %-6s %.3f ms/iter core_rows=%d hash=%016llx finite=%d\n", mode.c_str(),
                    wall_ms / static_cast<double>(iterations), core_rows, static_cast<unsigned long long>(hash),
                    finite ? 1 : 0);
    }
    return finite ? 0 : 1;
}
