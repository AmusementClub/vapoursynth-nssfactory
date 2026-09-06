#include "nss/cpu_api.hpp"
#include "nss/cpu_batch.hpp"
#include "nss/cpu_common.hpp"
#include "nss/cpu_lssc.hpp"
#include "nss/cpu_mcwnnm.hpp"
#include "nss/cpu_ncsr.hpp"
#include "nss/cpu_nlh.hpp"
#include "nss/cpu_twsc.hpp"
#include "nss/params.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#ifndef NSS_VERSION_STRING
#define NSS_VERSION_STRING "unknown"
#endif

namespace {

using clock = std::chrono::steady_clock;

struct Plane {
    int w = 0;
    int h = 0;
    int stride = 0;
    std::vector<float> buf;
    float* ptr() { return buf.data(); }
    const float* ptr() const { return buf.data(); }
};

struct BenchResult {
    std::string name;
    std::string group;
    double ms = 0.0;
    int iterations = 0;
    int width = 0;
    int height = 0;
};

std::string json_escape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (const unsigned char ch : value) {
        switch (ch) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (ch < 0x20) {
                out += '?';
            } else {
                out += static_cast<char>(ch);
            }
            break;
        }
    }
    return out;
}

std::string cpu_description() {
    if (const char* override_name = std::getenv("NSS_BENCH_CPU"); override_name && *override_name) {
        return override_name;
    }
#if defined(__linux__)
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        constexpr const char* kPrefix = "model name\t: ";
        if (line.rfind(kPrefix, 0) == 0) {
            return line.substr(std::char_traits<char>::length(kPrefix));
        }
    }
#endif
#if defined(__aarch64__) || defined(__arm64__)
    return "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#else
    return "unknown";
#endif
}

std::string compiler_description() {
#if defined(__clang__)
    return std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
    return std::string("gcc ") + __VERSION__;
#elif defined(_MSC_VER)
    return std::string("msvc ") + std::to_string(_MSC_VER);
#else
    return "unknown";
#endif
}

int benchmark_thread_count() {
    if (const char* value = std::getenv("NSS_BENCH_THREADS"); value && *value) {
        char* end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end != value && *end == '\0' && parsed > 0 && parsed < 100000) {
            return static_cast<int>(parsed);
        }
    }
    return 1;
}

unsigned benchmark_seed() {
    if (const char* value = std::getenv("NSS_BENCH_SEED"); value && *value) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(value, &end, 10);
        if (end != value && *end == '\0' && parsed <= 0xffffffffUL) {
            return static_cast<unsigned>(parsed);
        }
    }
    return 42u;
}

unsigned benchmark_frame() {
    if (const char* value = std::getenv("NSS_BENCH_FRAME"); value && *value) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(value, &end, 10);
        if (end != value && *end == '\0' && parsed <= 0xffffffffUL) {
            return static_cast<unsigned>(parsed);
        }
    }
    return 0u;
}

Plane make_plane(int w, int h, unsigned seed) {
    Plane p;
    p.w = w;
    p.h = h;
    p.stride = (w + 15) & ~15;
    p.buf.assign(static_cast<std::size_t>(p.stride) * static_cast<std::size_t>(h), 0.f);
    std::uint32_t s = seed ? seed : 1u;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            s = s * 1664525u + 1013904223u;
            const float n = static_cast<float>(s >> 8) * (1.0f / 16777216.0f);
            p.buf[static_cast<std::size_t>(y * p.stride + x)] = 0.35f + 0.2f * n;
        }
    }
    return p;
}

double ms_since(clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

void pin_busy() {
    // Touch a byte so the compiler cannot DCE the kernel.
}

}  // namespace

static double bench_ssd(Plane& a, int iters) {
    const int block = 8;
    volatile float sink = 0.f;
    const auto t0 = clock::now();
    for (int i = 0; i < iters; ++i) {
        const int x = (i * 3) % (a.w - block);
        const int y = (i * 5) % (a.h - block);
        sink += nss::ssd_block(a.ptr() + y * a.stride + x, a.stride,
                               a.ptr() + (y + 1) * a.stride + x + 1, a.stride, block);
    }
    (void)sink;
    return ms_since(t0);
}

static double bench_match(Plane& a, int iters) {
    nss::Match m[nss::kBmMaxGroup];
    volatile int sink = 0;
    const auto t0 = clock::now();
    for (int i = 0; i < iters; ++i) {
        const int bx = (i * 8) % (a.w - 8);
        const int by = (i * 8) % (a.h - 8);
        sink += nss::spatial_match(a.ptr(), a.stride, a.w, a.h, bx, by, 8, 7, 8, m);
    }
    (void)sink;
    return ms_since(t0);
}

static double bench_match_batch_shape(Plane& a, int block, int step, int group, int range, int iters) {
    constexpr int count = 32;
    std::array<nss::MatchBatchItem, count> items{};
    std::array<nss::Match, count * nss::kBmMaxGroup> matches{};
    std::array<int, count> counts{};
    for (int i = 0; i < count; ++i) {
        items[static_cast<std::size_t>(i)] =
            nss::MatchBatchItem{range + (i % 8) * step, range + (i / 8) * step, block, range, group};
    }
    volatile int sink = 0;
    const auto t0 = clock::now();
    for (int i = 0; i < iters; ++i) {
        sink += nss::spatial_match_batch(a.ptr(), a.stride, a.w, a.h, items.data(), count, matches.data(),
                                         nss::kBmMaxGroup, counts.data());
        sink += counts[static_cast<std::size_t>(i % count)];
    }
    (void)sink;
    return ms_since(t0);
}

static double bench_predictive_batch(Plane& p0, Plane& p1, Plane& p2, int radius, int iters) {
    constexpr int count = 32;
    constexpr int max_frames = nss::kBmMaxRadius * 2 + 1;
    std::array<const float*, max_frames> refs{};
    std::array<int, max_frames> strides{};
    const Plane* sources[3] = {&p0, &p1, &p2};
    const int ntemp = radius * 2 + 1;
    for (int t = 0; t < ntemp; ++t) {
        refs[static_cast<std::size_t>(t)] = sources[t % 3]->ptr();
        strides[static_cast<std::size_t>(t)] = sources[t % 3]->stride;
    }
    nss::SearchConfig cfg;
    cfg.block = 8;
    cfg.step = 8;
    cfg.group = 8;
    cfg.bm_range = 7;
    cfg.radius = radius;
    cfg.ps_num = 2;
    cfg.ps_range = 7;
    std::array<nss::MatchBatchItem, count> items{};
    std::array<nss::Match, count * nss::kBmMaxGroup> matches{};
    std::array<int, count> counts{};
    for (int i = 0; i < count; ++i) {
        items[static_cast<std::size_t>(i)] = nss::MatchBatchItem{7 + (i % 8) * 8, 7 + (i / 8) * 8, 8, 7, 8};
    }
    volatile int sink = 0;
    const auto t0 = clock::now();
    for (int i = 0; i < iters; ++i) {
        sink += nss::predictive_match_batch(refs.data(), strides.data(), ntemp, p0.w, p0.h, radius, cfg,
                                            items.data(), count, matches.data(), nss::kBmMaxGroup, counts.data());
        sink += counts[static_cast<std::size_t>(i % count)];
    }
    (void)sink;
    return ms_since(t0);
}

static double bench_pack_unpack(Plane& a, int iters) {
    const int block = 8;
    float patch[64];
    std::vector<float> num(static_cast<std::size_t>(a.stride) * a.h, 0.f);
    std::vector<float> den(num.size(), 0.f);
    const auto t0 = clock::now();
    for (int i = 0; i < iters; ++i) {
        const int x = (i * 8) % (a.w - block);
        const int y = (i * 8) % (a.h - block);
        nss::pack_patch(patch, 64, a.ptr(), a.stride, x, y, block, a.w, a.h);
        nss::unpack_patch(num.data(), den.data(), a.stride, x, y, patch, block, a.w, a.h, 1.f);
    }
    pin_busy();
    return ms_since(t0);
}

static double bench_dct(int iters) {
    float block[64];
    for (int i = 0; i < 64; ++i) {
        block[i] = 0.01f * static_cast<float>(i);
    }
    const auto t0 = clock::now();
    for (int i = 0; i < iters; ++i) {
        nss::dct8_2d(block);
        nss::idct8_2d(block);
    }
    volatile float sink = block[0];
    (void)sink;
    return ms_since(t0);
}

static double bench_bm3d_group(int iters) {
    const int group = 8;
    const int block = 8;
    const int area = 64;
    std::vector<float> patches(static_cast<std::size_t>(group * area), 0.1f);
    std::vector<float> work(static_cast<std::size_t>(nss::bm3d_filter_work_floats(group, block)));
    float w = 1.f;
    const auto t0 = clock::now();
    for (int i = 0; i < iters; ++i) {
        nss::bm3d_filter_group(patches.data(), area, group, group, block, 3.f / 255.f, false, nullptr, &w,
                               work.data());
    }
    volatile float sink = patches[0] + w;
    (void)sink;
    return ms_since(t0);
}

static double bench_bm3d_frame(Plane& src, int iters) {
    const int block = 8;
    const int group = 8;
    const int step = 8;
    std::vector<float> num(static_cast<std::size_t>(src.w) * src.h, 0.f);
    std::vector<float> den(num.size(), 0.f);
    std::vector<float> dst(static_cast<std::size_t>(src.stride) * src.h, 0.f);
    nss::Match matches[nss::kBmMaxGroup];
    const auto t0 = clock::now();
    for (int it = 0; it < iters; ++it) {
        std::fill(num.begin(), num.end(), 0.f);
        std::fill(den.begin(), den.end(), 0.f);
        for (int by0 = 0; by0 < src.h - block + step; by0 += step) {
            const int by = std::min(by0, src.h - block);
            for (int bx0 = 0; bx0 < src.w - block + step; bx0 += step) {
                const int bx = std::min(bx0, src.w - block);
                const int k = nss::spatial_match(src.ptr(), src.stride, src.w, src.h, bx, by, block, 7, group, matches);
                if (k <= 0) {
                    continue;
                }
                nss::bm3d_filter8(src.ptr(), src.stride, matches, k, 3.f / 255.f, false, nullptr, src.stride, num.data(),
                                  den.data(), src.w, src.w, src.h);
            }
        }
        nss::aggregate_finish(dst.data(), num.data(), den.data(), src.ptr(), src.w, src.h, src.stride, src.w);
    }
    volatile float sink = dst[static_cast<std::size_t>(src.h / 2 * src.stride + src.w / 2)];
    (void)sink;
    return ms_since(t0);
}

static double bench_nlm(Plane& src, int iters) {
    const int w = src.w;
    const int h = src.h;
    const int stride = src.stride;
    const int size = h * stride;
    std::vector<float> workspace(static_cast<std::size_t>(size) * 5 + static_cast<std::size_t>(w), 0.f);
    float* weightp = workspace.data();
    float* wdstp = weightp + size;
    float* maxw = wdstp + size;
    float* temp = maxw + size;
    float* temp_bwd = temp + size;
    float* buffer = temp_bwd + size;
    std::vector<float> dst(static_cast<std::size_t>(size), 0.f);
    const int a = 2;
    const int s = 4;
    const float h2 = (255.0f * 255.0f) / (3.0f * 1.2f * 1.2f * 81.0f);
    const auto t0 = clock::now();
    for (int it = 0; it < iters; ++it) {
        std::fill(weightp, weightp + size, 0.f);
        std::fill(wdstp, wdstp + size, 0.f);
        std::fill(maxw, maxw + size, 1e-12f);
        for (int oy = -a; oy <= a; ++oy) {
            for (int ox = -a; ox <= a; ++ox) {
                if (oy * (2 * a + 1) + ox >= 0) {
                    continue;
                }
                nss::nlm_distance_luma_f32(temp_bwd, src.ptr(), src.ptr(), ox, oy, w, h, stride);
                nss::nlm_horizontal(temp, temp_bwd, s, w, h, stride);
                nss::nlm_vertical_welsch(temp_bwd, temp, s, h2, w, h, stride, buffer);
                nss::nlm_accum_ch1(weightp, wdstp, maxw, src.ptr(), src.ptr(), temp_bwd, temp_bwd, ox, oy, w, h, stride);
            }
        }
        nss::nlm_finish_ch1(dst.data(), src.ptr(), weightp, wdstp, maxw, 1.0f, w, h, stride);
    }
    volatile float sink = dst[static_cast<std::size_t>(h / 2 * stride + w / 2)];
    (void)sink;
    return ms_since(t0);
}

static double bench_wnnm_frame(Plane& src, int iters) {
    const int block = 8;
    const int group = 8;
    const int step = 8;
    const int m = block * block;
    const int lda = m;
    std::vector<float> num(static_cast<std::size_t>(src.w) * src.h, 0.f);
    std::vector<float> den(num.size(), 0.f);
    std::vector<float> dst(static_cast<std::size_t>(src.stride) * src.h, 0.f);
    std::vector<float> patches(static_cast<std::size_t>(lda * group));
    std::vector<float> work(static_cast<std::size_t>(nss::wnnm_shrink_work_floats(m, group)));
    nss::Match matches[nss::kWnnmMaxGroup];
    const float sigma = 3.f / 255.f;
    const auto t0 = clock::now();
    for (int it = 0; it < iters; ++it) {
        std::fill(num.begin(), num.end(), 0.f);
        std::fill(den.begin(), den.end(), 0.f);
        for (int by0 = 0; by0 < src.h - block + step; by0 += step) {
            const int by = std::min(by0, src.h - block);
            for (int bx0 = 0; bx0 < src.w - block + step; bx0 += step) {
                const int bx = std::min(bx0, src.w - block);
                const int k = nss::spatial_match(src.ptr(), src.stride, src.w, src.h, bx, by, block, 7, group, matches);
                if (k <= 0) {
                    continue;
                }
                for (int i = 0; i < k; ++i) {
                    nss::pack_patch(patches.data() + i * lda, lda, src.ptr(), src.stride, matches[i].x, matches[i].y,
                                    block, src.w, src.h);
                }
                float aw = 1.f;
                if (nss::wnnm_shrink(patches.data(), m, k, lda, sigma, 0, 1, &aw, work.data(),
                                     static_cast<int>(work.size())) != 0) {
                    continue;
                }
                for (int i = 0; i < k; ++i) {
                    nss::aggregate_add(num.data(), den.data(), src.w, matches[i].x, matches[i].y,
                                       patches.data() + i * lda, block, src.w, src.h, aw);
                }
            }
        }
        nss::aggregate_finish(dst.data(), num.data(), den.data(), src.ptr(), src.w, src.h, src.stride, src.w);
    }
    volatile float sink = dst[static_cast<std::size_t>(src.h / 2 * src.stride + src.w / 2)];
    (void)sink;
    return ms_since(t0);
}

static double bench_nlh_frame(Plane& src, int iters) {
    const int block = nss::kNlhDefaultBlock;
    const int group = nss::kNlhDefaultGroup;
    const int step = nss::kNlhDefaultStep;
    const int range = nss::kNlhDefaultRange;
    const int q = nss::kNlhDefaultQ;
    const int m = block * block;
    const int lda = (m + 15) & ~15;
    const float sigma = nss::kNlhDefaultSigma / 255.f;
    std::vector<float> num(static_cast<std::size_t>(src.w) * src.h, 0.f);
    std::vector<float> den(num.size(), 0.f);
    std::vector<float> dst(static_cast<std::size_t>(src.stride) * src.h, 0.f);
    std::vector<float> patches(static_cast<std::size_t>(lda * group));
    std::vector<float> work(static_cast<std::size_t>(nss::nlh_filter_work_floats(m, group, q, lda)));
    nss::Match matches[nss::kBmMaxGroup];
    const auto t0 = clock::now();
    for (int it = 0; it < iters; ++it) {
        std::fill(num.begin(), num.end(), 0.f);
        std::fill(den.begin(), den.end(), 0.f);
        for (int by0 = 0; by0 < src.h - block + step; by0 += step) {
            const int by = std::min(by0, src.h - block);
            for (int bx0 = 0; bx0 < src.w - block + step; bx0 += step) {
                const int bx = std::min(bx0, src.w - block);
                const int k = nss::spatial_match(src.ptr(), src.stride, src.w, src.h, bx, by, block, range, group,
                                                 matches);
                if (k <= 0) {
                    continue;
                }
                for (int i = 0; i < k; ++i) {
                    nss::pack_patch(patches.data() + i * lda, lda, src.ptr(), src.stride, matches[i].x, matches[i].y,
                                    block, src.w, src.h);
                }
                float aw = 1.f;
                nss::nlh_filter_group(patches.data(), m, k, lda, q, sigma, false, nullptr, &aw, work.data(),
                                      static_cast<int>(work.size()));
                for (int i = 0; i < k; ++i) {
                    nss::aggregate_add(num.data(), den.data(), src.w, matches[i].x, matches[i].y,
                                       patches.data() + i * lda, block, src.w, src.h, aw);
                }
            }
        }
        nss::aggregate_finish(dst.data(), num.data(), den.data(), src.ptr(), src.w, src.h, src.stride, src.w);
    }
    volatile float sink = dst[static_cast<std::size_t>(src.h / 2 * src.stride + src.w / 2)];
    (void)sink;
    return ms_since(t0);
}

static double bench_mcwnnm_frame(Plane& p0, Plane& p1, Plane& p2, int iters) {
    const int block = nss::kMcwnnmDefaultBlock;
    const int group = nss::kMcwnnmDefaultGroup;
    const int step = nss::kMcwnnmDefaultStep;
    const int range = nss::kMcwnnmDefaultRange;
    const int nch = 3;
    const int m = nch * block * block;
    const int lda = (m + 15) & ~15;
    const int w = p0.w;
    const int h = p0.h;
    const float sig[3] = {nss::kMcwnnmDefaultSigma / 255.f, nss::kMcwnnmDefaultSigma / 255.f,
                          nss::kMcwnnmDefaultSigma / 255.f};
    const float* refs[3] = {p0.ptr(), p1.ptr(), p2.ptr()};
    const int strides[3] = {p0.stride, p1.stride, p2.stride};
    std::vector<float> num(static_cast<std::size_t>(nch) * w * h, 0.f);
    std::vector<float> den(num.size(), 0.f);
    std::vector<float> patches(static_cast<std::size_t>(lda * group));
    std::vector<float> work(static_cast<std::size_t>(nss::mcwnnm_filter_work_floats(m, group)));
    nss::Match matches[nss::kBmMaxGroup];
    const auto t0 = clock::now();
    for (int it = 0; it < iters; ++it) {
        std::fill(num.begin(), num.end(), 0.f);
        std::fill(den.begin(), den.end(), 0.f);
        for (int by0 = 0; by0 < h - block + step; by0 += step) {
            const int by = std::min(by0, h - block);
            for (int bx0 = 0; bx0 < w - block + step; bx0 += step) {
                const int bx = std::min(bx0, w - block);
                const int k = nss::spatial_match_nch(refs, strides, nch, w, h, bx, by, block, range, group, matches);
                if (k <= 0) {
                    continue;
                }
                for (int i = 0; i < k; ++i) {
                    nss::pack_patch_nch(patches.data() + i * lda, lda, refs, strides, nch, matches[i].x, matches[i].y,
                                        block, w, h);
                }
                float aw = 1.f;
                if (nss::mcwnnm_filter_group(patches.data(), m, k, lda, nch, sig, nss::kMcwnnmDefaultAdmmIter,
                                             nss::kMcwnnmDefaultRho, nss::kMcwnnmDefaultMu,
                                             nss::kMcwnnmDefaultResidual, nss::kMcwnnmDefaultAdaptive, &aw,
                                             work.data(), static_cast<int>(work.size())) != 0) {
                    continue;
                }
                float* nums[3] = {num.data(), num.data() + w * h, num.data() + 2 * w * h};
                float* dens[3] = {den.data(), den.data() + w * h, den.data() + 2 * w * h};
                const int agg_st[3] = {w, w, w};
                for (int i = 0; i < k; ++i) {
                    nss::unpack_patch_nch(nums, dens, agg_st, nch, matches[i].x, matches[i].y, patches.data() + i * lda,
                                          block, w, h, aw);
                }
            }
        }
    }
    volatile float sink = num[static_cast<std::size_t>(h / 2 * w + w / 2)];
    (void)sink;
    return ms_since(t0);
}

static double bench_twsc_frame(Plane& src, int iters) {
    const int block = nss::kTwscDefaultBlock;
    const int group = nss::kTwscDefaultGroup;
    const int step = nss::kTwscDefaultStep;
    const int range = nss::kTwscDefaultRange;
    const int m = block * block;
    const int lda = (m + 15) & ~15;
    const float sigma = nss::kTwscDefaultSigma / 255.f;
    std::vector<float> num(static_cast<std::size_t>(src.w) * src.h, 0.f);
    std::vector<float> den(num.size(), 0.f);
    std::vector<float> dst(static_cast<std::size_t>(src.stride) * src.h, 0.f);
    std::vector<float> patches(static_cast<std::size_t>(lda * group));
    std::vector<float> work(static_cast<std::size_t>(nss::twsc_pca_soft_work_floats(m, group)));
    nss::Match matches[nss::kBmMaxGroup];
    const auto t0 = clock::now();
    for (int it = 0; it < iters; ++it) {
        std::fill(num.begin(), num.end(), 0.f);
        std::fill(den.begin(), den.end(), 0.f);
        for (int by0 = 0; by0 < src.h - block + step; by0 += step) {
            const int by = std::min(by0, src.h - block);
            for (int bx0 = 0; bx0 < src.w - block + step; bx0 += step) {
                const int bx = std::min(bx0, src.w - block);
                const int k = nss::spatial_match(src.ptr(), src.stride, src.w, src.h, bx, by, block, range, group,
                                                 matches);
                if (k <= 0) {
                    continue;
                }
                for (int i = 0; i < k; ++i) {
                    nss::pack_patch(patches.data() + i * lda, lda, src.ptr(), src.stride, matches[i].x, matches[i].y,
                                    block, src.w, src.h);
                }
                if (nss::twsc_pca_soft(patches.data(), m, k, lda, sigma, work.data(), static_cast<int>(work.size())) !=
                    0) {
                    continue;
                }
                for (int i = 0; i < k; ++i) {
                    nss::aggregate_add(num.data(), den.data(), src.w, matches[i].x, matches[i].y,
                                       patches.data() + i * lda, block, src.w, src.h, 1.f);
                }
            }
        }
        nss::aggregate_finish(dst.data(), num.data(), den.data(), src.ptr(), src.w, src.h, src.stride, src.w);
    }
    volatile float sink = dst[static_cast<std::size_t>(src.h / 2 * src.stride + src.w / 2)];
    (void)sink;
    return ms_since(t0);
}

static double bench_ncsr_frame(Plane& src, int iters) {
    const int block = nss::kNcsrDefaultBlock;
    const int group = nss::kNcsrDefaultGroup;
    const int step = nss::kNcsrDefaultStep;
    const int range = nss::kNcsrDefaultRange;
    const float sigma = nss::kNcsrDefaultSigma / 255.f;
    std::vector<float> dst(static_cast<std::size_t>(src.stride) * src.h, 0.f);
    const int need = nss::ncsr_denoise_work_floats(src.w, src.h, block, group);
    std::vector<float> work(static_cast<std::size_t>(need));
    const auto t0 = clock::now();
    for (int it = 0; it < iters; ++it) {
        nss::ncsr_denoise_plane(src.ptr(), src.w, src.h, src.stride, dst.data(), src.stride, block, step, group, range,
                                sigma, nss::kNcsrDefaultIters, nss::kNcsrDefaultDelta, work.data(), need);
    }
    volatile float sink = dst[static_cast<std::size_t>(src.h / 2 * src.stride + src.w / 2)];
    (void)sink;
    return ms_since(t0);
}

static double bench_lssc_frame(Plane& src, int iters) {
    const int block = nss::kLsscDefaultBlock;
    const int step = nss::kLsscDefaultStep;
    const float sigma = nss::kLsscDefaultSigma / 255.f;
    std::vector<float> num(static_cast<std::size_t>(src.w) * src.h, 0.f);
    std::vector<float> den(num.size(), 0.f);
    const int need = nss::lssc_denoise_work_floats(src.w, src.h, block, step);
    std::vector<float> work(static_cast<std::size_t>(need));
    const auto t0 = clock::now();
    for (int it = 0; it < iters; ++it) {
        nss::lssc_denoise_plane(src.ptr(), src.w, src.h, src.stride, num.data(), den.data(), src.w, block, step, sigma,
                                work.data(), need);
    }
    volatile float sink = num[static_cast<std::size_t>(src.h / 2 * src.w + src.w / 2)];
    (void)sink;
    return ms_since(t0);
}

static double bench_nlh_pixel(int iters) {
    const int m = 64;
    const int n = 16;
    const int lda = 64;
    const int q = 4;
    std::vector<float> group(static_cast<std::size_t>(lda * n));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            group[static_cast<std::size_t>(i + j * lda)] = 0.01f * static_cast<float>(i) + 0.03f * static_cast<float>(j);
        }
    }
    std::vector<int> idx(static_cast<std::size_t>(m * q));
    const auto t0 = clock::now();
    for (int i = 0; i < iters; ++i) {
        nss::pixel_match(group.data(), m, n, lda, q, idx.data());
        group[0] += 1e-8f * static_cast<float>(idx[0]);
    }
    volatile int sink = idx[0];
    (void)sink;
    return ms_since(t0);
}

static double bench_nlh_group(int iters) {
    const int m = 64;
    const int n = 16;
    const int lda = 64;
    const int q = 4;
    std::vector<float> patches(static_cast<std::size_t>(lda * n), 0.1f);
    std::vector<float> work(static_cast<std::size_t>(nss::nlh_filter_work_floats(m, n, q, lda)));
    float aw = 1.f;
    const auto t0 = clock::now();
    for (int i = 0; i < iters; ++i) {
        nss::nlh_filter_group(patches.data(), m, n, lda, q, 3.f / 255.f, false, nullptr, &aw, work.data(),
                              static_cast<int>(work.size()));
    }
    volatile float sink = patches[0] + aw;
    (void)sink;
    return ms_since(t0);
}

static double bench_mcwnnm_group(int iters) {
    const int nch = 3;
    const int m = nch * 64;
    const int n = 8;
    const int lda = m;
    std::vector<float> Y(static_cast<std::size_t>(lda * n));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            Y[static_cast<std::size_t>(i + j * lda)] = 0.2f + 0.001f * static_cast<float>(i + j);
        }
    }
    std::vector<float> work(static_cast<std::size_t>(nss::mcwnnm_filter_work_floats(m, n)));
    const float sig[3] = {3.f / 255.f, 3.f / 255.f, 3.f / 255.f};
    const auto t0 = clock::now();
    for (int i = 0; i < iters; ++i) {
        nss::mcwnnm_filter_group(Y.data(), m, n, lda, nch, sig, nss::kMcwnnmDefaultAdmmIter, nss::kMcwnnmDefaultRho,
                                 nss::kMcwnnmDefaultMu, nss::kMcwnnmDefaultResidual, nss::kMcwnnmDefaultAdaptive,
                                 nullptr, work.data(), static_cast<int>(work.size()));
    }
    volatile float sink = Y[0];
    (void)sink;
    return ms_since(t0);
}

static double bench_twsc_group(int iters) {
    const int m = 64;
    const int n = 8;
    const int lda = 64;
    std::vector<float> Y(static_cast<std::size_t>(lda * n), 0.15f);
    for (int j = 0; j < n; ++j) {
        Y[static_cast<std::size_t>(j * lda)] += 0.01f * static_cast<float>(j);
    }
    std::vector<float> work(static_cast<std::size_t>(nss::twsc_pca_soft_work_floats(m, n)));
    const auto t0 = clock::now();
    for (int i = 0; i < iters; ++i) {
        nss::twsc_pca_soft(Y.data(), m, n, lda, 3.f / 255.f, work.data(), static_cast<int>(work.size()));
    }
    volatile float sink = Y[0];
    (void)sink;
    return ms_since(t0);
}

static double bench_ncsr_group(int iters) {
    const int m = 64;
    const int n = 8;
    const int lda = 64;
    std::vector<float> Y(static_cast<std::size_t>(lda * n), 0.15f);
    std::vector<float> work(static_cast<std::size_t>(nss::ncsr_filter_work_floats(m, n)));
    const auto t0 = clock::now();
    for (int i = 0; i < iters; ++i) {
        nss::ncsr_filter_group(Y.data(), m, n, lda, 3.f / 255.f, nullptr, work.data(), static_cast<int>(work.size()));
    }
    volatile float sink = Y[0];
    (void)sink;
    return ms_since(t0);
}

static double bench_lssc_ista(int iters) {
    const int m = 64;
    const int n = 32;
    const int atoms = 64;
    const int lda = m;
    const int ldd = m;
    std::vector<float> patches(static_cast<std::size_t>(lda * n), 0.2f);
    std::vector<float> D(static_cast<std::size_t>(ldd * atoms), 0.f);
    nss::lssc_dict_init(D.data(), m, atoms, ldd, patches.data(), n, lda, 8, 0, 1u);
    std::vector<float> work(static_cast<std::size_t>(nss::lssc_reconstruct_work_floats(m, n, atoms)));
    const auto t0 = clock::now();
    for (int i = 0; i < iters; ++i) {
        nss::lssc_reconstruct(patches.data(), m, n, lda, D.data(), atoms, ldd, 3.f / 255.f, work.data(),
                              static_cast<int>(work.size()));
    }
    volatile float sink = patches[0];
    (void)sink;
    return ms_since(t0);
}

static double bench_svd(int m, int n, int iters) {
    const int lda = m;
    std::vector<float> A(static_cast<std::size_t>(lda * n));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            A[static_cast<std::size_t>(i + j * lda)] = 0.01f * static_cast<float>(i + 3 * j);
        }
    }
    std::vector<float> U(static_cast<std::size_t>(m * n));
    std::vector<float> S(static_cast<std::size_t>(n));
    std::vector<float> Vt(static_cast<std::size_t>(n * n));
    const int wneed = m * n * 6 + n * n * 8 + n + 256;
    std::vector<float> work(static_cast<std::size_t>(wneed));
    const auto t0 = clock::now();
    for (int i = 0; i < iters; ++i) {
        nss::svd_economy(m, n, A.data(), lda, U.data(), m, S.data(), Vt.data(), n, work.data(), wneed);
    }
    volatile float sink = S[0];
    (void)sink;
    return ms_since(t0);
}

static void print_ms(const char* name, double ms, int iters, int w = 0, int h = 0) {
    if (w > 0 && h > 0 && iters > 0) {
        const double mpix_s = (static_cast<double>(w) * h * iters / 1e6) / (ms / 1000.0);
        std::printf("%-18s  %8.2f ms  (%d iter, %8.3f ms/iter, %6.3f MPix/s)\n", name, ms, iters,
                    ms / static_cast<double>(iters), mpix_s);
    } else {
        std::printf("%-18s  %8.2f ms  (%d iter, %8.3f ms/iter)\n", name, ms, iters, ms / static_cast<double>(iters));
    }
}

int main(int argc, char** argv) {
    bool json = std::getenv("NSS_BENCH_JSON") != nullptr;
    std::vector<const char*> positional;
    positional.reserve(static_cast<std::size_t>(argc));
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--json") == 0) {
            json = true;
        } else {
            positional.push_back(argv[i]);
        }
    }
    std::string which = positional.empty() ? "wave2" : positional[0];
    const bool wave2_default = (which == "wave2" || which == "kernels" || which == "nlh" || which == "mcwnnm" ||
                                which == "twsc" || which == "ncsr" || which == "lssc" || which == "nlh_pixel" ||
                                which == "nlh_group" || which == "mcwnnm_group" || which == "twsc_group" ||
                                which == "ncsr_group" || which == "lssc_ista" || which == "svd64" || which == "svd192");
    const int w = (positional.size() > 1) ? std::atoi(positional[1]) : (wave2_default ? 256 : 1280);
    const int h = (positional.size() > 2) ? std::atoi(positional[2]) : (wave2_default ? 256 : 720);
    const int frame_iters = (positional.size() > 3) ? std::max(1, std::atoi(positional[3])) : 1;
    const bool warmup = std::getenv("NSS_NO_WARMUP") == nullptr;
    const int thread_count = benchmark_thread_count();
    const int iteration_multiplier = []() {
        const char* value = std::getenv("NSS_BENCH_ITERS_MULTIPLIER");
        return value ? std::max(1, std::atoi(value)) : 1;
    }();
    const std::string cpu = cpu_description();
    const std::string compiler = compiler_description();
    const std::string revision = NSS_VERSION_STRING;
    const unsigned seed = benchmark_seed();
    const unsigned frame = benchmark_frame();
    // Keep frame number in the generated content so paired runs exercise
    // distinct frame states while remaining deterministic for each pair.
    const unsigned content_seed = seed + frame * 2654435761u;
    Plane src = make_plane(w, h, content_seed);
    Plane p1 = make_plane(w, h, content_seed + 1u);
    Plane p2 = make_plane(w, h, content_seed + 2u);
    if (!json) {
        std::printf("plane %dx%d stride=%d which=%s warmup=%d\n", w, h, src.stride, which.c_str(), warmup ? 1 : 0);
    }
    std::vector<BenchResult> results;

    auto want = [&](const char* name, const char* group) {
        return which == "all" || which == name || (group && which == group);
    };
    auto run = [&](const char* name, const char* group, auto fn, int iters, int pw = 0, int ph = 0) {
        if (!want(name, group)) {
            return;
        }
        if (warmup) {
            fn(1);
        }
        iters *= iteration_multiplier;
        const double ms = fn(iters);
        if (json) {
            results.push_back(BenchResult{std::string(name), std::string(group ? group : ""), ms, iters, pw, ph});
        } else {
            print_ms(name, ms, iters, pw, ph);
            std::fflush(stdout);
        }
    };

    run("ssd", "wave1", [&](int n) { return bench_ssd(src, n); }, 200000);
    run("match", "wave1", [&](int n) { return bench_match(src, n); }, 4000);
    run("match_batch8", nullptr, [&](int n) { return bench_match_batch_shape(src, 8, 8, 8, 7, n); }, 1200);
    run("match_batch8_g3", nullptr, [&](int n) { return bench_match_batch_shape(src, 8, 8, 3, 7, n); }, 1200);
    run("match_batch4", nullptr, [&](int n) { return bench_match_batch_shape(src, 4, 4, 8, 7, n); }, 500);
    run("predictive_batch_r1", nullptr, [&](int n) { return bench_predictive_batch(src, p1, p2, 1, n); }, 300);
    run("predictive_batch_r4", nullptr, [&](int n) { return bench_predictive_batch(src, p1, p2, 4, n); }, 100);
    run("pack", "wave1", [&](int n) { return bench_pack_unpack(src, n); }, 20000);
    run("dct8", "wave1", [&](int n) { return bench_dct(n); }, 50000);
    run("bm3d_group", "wave1", [&](int n) { return bench_bm3d_group(n); }, 2000);
    run("bm3d", "wave1", [&](int n) { return bench_bm3d_frame(src, n); }, frame_iters, w, h);
    run("nlm", "wave1", [&](int n) { return bench_nlm(src, n); }, frame_iters, w, h);
    run("wnnm", "wave1", [&](int n) { return bench_wnnm_frame(src, n); }, frame_iters, w, h);

    run("nlh", "wave2", [&](int n) { return bench_nlh_frame(src, n); }, frame_iters, w, h);
    run("mcwnnm", "wave2", [&](int n) { return bench_mcwnnm_frame(src, p1, p2, n); }, frame_iters, w, h);
    run("twsc", "wave2", [&](int n) { return bench_twsc_frame(src, n); }, frame_iters, w, h);
    run("ncsr", "wave2", [&](int n) { return bench_ncsr_frame(src, n); }, frame_iters, w, h);
    run("lssc", "wave2", [&](int n) { return bench_lssc_frame(src, n); }, frame_iters, w, h);

    run("nlh_pixel", "kernels", [&](int n) { return bench_nlh_pixel(n); }, 2000);
    run("nlh_group", "kernels", [&](int n) { return bench_nlh_group(n); }, 400);
    run("mcwnnm_group", "kernels", [&](int n) { return bench_mcwnnm_group(n); }, 80);
    run("twsc_group", "kernels", [&](int n) { return bench_twsc_group(n); }, 400);
    run("ncsr_group", "kernels", [&](int n) { return bench_ncsr_group(n); }, 400);
    run("lssc_ista", "kernels", [&](int n) { return bench_lssc_ista(n); }, 80);
    run("svd64", "kernels", [&](int n) { return bench_svd(64, 8, n); }, 2000);
    run("svd192", "kernels", [&](int n) { return bench_svd(192, 8, n); }, 800);
    if (json) {
        std::printf("{\"schema\":\"nssfactory.bench.v2\",\"which\":\"%s\","
                    "\"git_revision\":\"%s\",\"compiler\":\"%s\",\"cpu\":\"%s\","
                    "\"width\":%d,\"height\":%d,\"input_shape\":\"%dx%d\","
                    "\"seed\":%u,\"frame_number\":%u,\"thread_count\":%d,\"warmup\":%s,"
                    "\"timing\":{\"clock\":\"steady_clock\",\"provenance\":\"process_wall_time\","
                    "\"worker_time_ms\":null},"
                    "\"filter_parameters\":{\"block\":8,\"step\":8,\"group\":8,"
                    "\"bm_range\":7,\"sigma\":3,\"nlm_d\":1,\"nlm_a\":2,\"nlm_s\":4,"
                    "\"nlm_h\":1.2,\"radius\":0,\"iterations\":%d},\"results\":[",
                    json_escape(which).c_str(), json_escape(revision).c_str(), json_escape(compiler).c_str(),
                    json_escape(cpu).c_str(), w, h, w, h, seed, frame, thread_count, warmup ? "true" : "false",
                    frame_iters);
        for (std::size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            if (i != 0) {
                std::printf(",");
            }
            std::printf("{\"name\":\"%s\",\"group\":\"%s\",\"wall_time_ms\":%.9g,"
                        "\"milliseconds\":%.9g,\"iterations\":%d,\"width\":%d,\"height\":%d}",
                        json_escape(r.name).c_str(), json_escape(r.group).c_str(), r.ms, r.ms, r.iterations, r.width,
                        r.height);
        }
        std::printf("]}\n");
    }
    return 0;
}
