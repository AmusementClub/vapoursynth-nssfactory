#include "cpu/bm/kernel_lab.hpp"
#include "hwy/targets.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#if defined(__unix__)
#include <sys/mman.h>
#include <unistd.h>
#endif

using Clock = std::chrono::steady_clock;

static nss::detail::SsdRowKernel ssd(int block, int variant) {
    return block == 12 ? nss::detail::ssd12_lab_kernel(variant) : nss::detail::ssd16_lab_kernel(variant);
}

static std::vector<float> noise(std::size_t count) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> data(count);
    for (auto& value : data) value = dist(rng);
    return data;
}

static void reference_dct2(std::vector<float>& data, int n, bool inverse) {
    constexpr double pi = 3.14159265358979323846;
    double table[16][16];
    for (int k = 0; k < n; ++k)
        for (int i = 0; i < n; ++i)
            table[k][i] = std::sqrt((k == 0 ? 1.0 : 2.0) / double(n)) * std::cos(pi * (i + .5) * k / double(n));
    for (std::size_t p = 0; p < data.size(); p += n * n) {
        double tmp[256];
        for (int r = 0; r < n; ++r)
            for (int c = 0; c < n; ++c) {
                double sum = 0;
                for (int i = 0; i < n; ++i)
                    sum += (inverse ? table[i][r] : table[r][i]) * data[p + i * n + c];
                tmp[r * n + c] = sum;
            }
        for (int r = 0; r < n; ++r)
            for (int c = 0; c < n; ++c) {
                double sum = 0;
                for (int i = 0; i < n; ++i)
                    sum += (inverse ? table[i][c] : table[c][i]) * tmp[r * n + i];
                data[p + r * n + c] = static_cast<float>(sum);
            }
    }
}

static int verify() {
    if (!ssd(12, 0) && !nss::detail::dct_lab_kernel(4, 0)) return 77;
    int failed = 0;
#if defined(__unix__)
    // End the final candidate patch at a PROT_NONE page. Masked/tail loads
    // must not read beyond the row even when no padding is available.
    const std::size_t page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    int guard_cases = 0;
    for (int b : {12, 16}) {
        if (!ssd(b, 0)) continue;
        for (int count : {1, 2, 7, 15, 16, 17, 127, 129}) {
            const int stride = count + b - 1;
            const std::size_t floats = static_cast<std::size_t>(stride) * b;
            const std::size_t pages = (floats * sizeof(float) + page - 1) / page;
            void* mapping = mmap(nullptr, (pages + 1) * page, PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (mapping == MAP_FAILED) return 1;
            char* boundary = static_cast<char*>(mapping) + pages * page;
            if (mprotect(boundary, page, PROT_NONE) != 0) return 1;
            float* source = reinterpret_cast<float*>(boundary) - floats;
            auto input = noise(floats);
            std::copy(input.begin(), input.end(), source);
            std::vector<float> reference(count), output(count);
            ssd(b, 0)(source, source, stride, count, reference.data());
            for (int v = 1; v < 3; ++v) {
                ++guard_cases;
                ssd(b, v)(source, source, stride, count, output.data());
                if (std::memcmp(reference.data(), output.data(), count * sizeof(float))) ++failed;
            }
            munmap(mapping, (pages + 1) * page);
        }
    }
    std::printf("{\"verify\":\"ssd_guard_pages\",\"cases\":%d,\"skipped\":%s,\"failed\":%d}\n",
                guard_cases, guard_cases ? "false" : "true", failed);
#endif
    for (int b : {12, 16}) {
        if (!ssd(b, 0)) continue;
        auto ref = ssd(b, 0);
        for (int v = 1; v < 3; ++v) {
            std::size_t different = 0;
            double max_abs = 0;
            for (int count = 1; count <= 129; ++count) {
                for (int offset : {0, 1, 7, 15}) {
                    const int stride = count + b + 19;
                    auto src = noise(static_cast<std::size_t>(stride) * (b + 1));
                    std::vector<float> a(count + 2, -12345.f), c = a;
                    ref(src.data() + 1, src.data() + offset, stride, count, a.data() + 1);
                    ssd(b, v)(src.data() + 1, src.data() + offset, stride, count, c.data() + 1);
                    if (c.front() != -12345.f || c.back() != -12345.f) ++failed;
                    for (int i = 1; i <= count; ++i) {
                        different += std::memcmp(&a[i], &c[i], sizeof(float)) != 0;
                        max_abs = std::max(max_abs, std::abs(double(a[i]) - c[i]));
                    }
                }
            }
            std::printf("{\"verify\":\"ssd\",\"block\":%d,\"variant\":%d,\"different\":%zu,\"max_abs\":%.9g}\n",
                        b, v, different, max_abs);
            if (different) ++failed;
        }
    }
    for (int block : {4, 8, 12, 16}) {
      auto baseline_kernel = nss::detail::dct_lab_kernel(block, 0);
      if (!baseline_kernel) { std::printf("{\"skip_block\":%d}\n", block); continue; }
      for (int v = 0; v < (block == 4 ? 1 : (block == 16 ? 5 : 3)); ++v) {
        auto kernel = nss::detail::dct_lab_kernel(block, v);
        if (!kernel) { ++failed; continue; }
        float invalid[256]{};
        if (kernel(nullptr, 1, false) || kernel(invalid, 0, false) || kernel(invalid, -1, true)) ++failed;
        std::size_t different = 0;
        double max_error = 0, roundtrip = 0;
        for (int count : {1, 2, 3, 7, 8, 9, 16, 32, 64}) {
          for (int pattern = 0; pattern < 5; ++pattern) {
            auto input = noise(static_cast<std::size_t>(count) * block * block);
            if (pattern == 1) std::fill(input.begin(), input.end(), 1.f);
            if (pattern == 2) { std::fill(input.begin(), input.end(), 0.f); input[block + 1] = 1.f; }
            if (pattern == 3)
              for (std::size_t i = 0; i < input.size(); ++i) input[i] = float((i / block) % block) / block;
            if (pattern == 4)
              for (std::size_t i = 0; i < input.size(); ++i) input[i] = i % 2 ? -1.f : 1.f;
            for (bool inverse : {false, true}) {
              auto baseline = input, reference = input;
              baseline_kernel(baseline.data(), count, inverse);
              reference_dct2(reference, block, inverse);
              for (int offset : {0, 1, 7, 15}) {
                std::vector<float> got(input.size() + offset + 16, -12345.f);
                std::copy(input.begin(), input.end(), got.begin() + offset);
                if (!kernel(got.data() + offset, count, inverse)) ++failed;
                for (std::size_t i = 0; i < got.size(); ++i) {
                  if (i < std::size_t(offset) || i >= input.size() + offset) {
                    if (got[i] != -12345.f) ++failed;
                  } else {
                    const auto j = i - offset;
                    if (block == 8 && v == 2 && different < 4 && std::memcmp(&baseline[j], &got[i], sizeof(float))) {
                        std::uint32_t a, b; std::memcpy(&a, &baseline[j], 4); std::memcpy(&b, &got[i], 4);
                        std::printf("{\"first_difference\":%zu,\"pattern\":%d,\"count\":%d,\"inverse\":%d,\"a\":%u,\"b\":%u}\n", j, pattern, count, inverse, a, b);
                    }
                    different += std::memcmp(&baseline[j], &got[i], sizeof(float)) != 0;
                    if (!std::isfinite(got[i])) ++failed;
                    max_error = std::max(max_error, std::abs(double(reference[j]) - got[i]));
                  }
                }
              }
#if defined(__unix__)
              const std::size_t pages = (input.size() * sizeof(float) + page - 1) / page;
              void* mapping = mmap(nullptr, (pages + 1) * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
              if (mapping == MAP_FAILED) return 1;
              char* boundary = static_cast<char*>(mapping) + pages * page;
              if (mprotect(boundary, page, PROT_NONE)) return 1;
              float* tail = reinterpret_cast<float*>(boundary) - input.size();
              std::copy(input.begin(), input.end(), tail);
              if (!kernel(tail, count, inverse)) ++failed;
              different += std::memcmp(tail, baseline.data(), input.size() * sizeof(float)) != 0;
              munmap(mapping, (pages + 1) * page);
#endif
            }
            auto got = input;
            kernel(got.data(), count, false);
            kernel(got.data(), count, true);
            for (std::size_t i = 0; i < got.size(); ++i) {
              if (!std::isfinite(got[i])) ++failed;
              roundtrip = std::max(roundtrip, std::abs(double(input[i]) - got[i]));
            }
          }
        }
        std::printf("{\"verify\":\"dct\",\"block\":%d,\"variant\":%d,\"different\":%zu,\"reference_max\":%.9g,\"roundtrip_max\":%.9g}\n",
                    block, v, different, max_error, roundtrip);
        if (different || max_error > 2e-5 || roundtrip > 2e-5) ++failed;
      }
    }
    return failed ? 1 : 0;
}

int main(int argc, char** argv) {
    if (const char* target = std::getenv("NSS_LAB_TARGET")) {
        const std::string name(target);
        const auto supported = hwy::SupportedTargets();
        const auto requested = name == "AVX2" ? HWY_AVX2 : (name == "AVX3" ? HWY_AVX3 : 0);
        if (!requested) return 2;
        if (!(supported & requested)) return 77;
        hwy::SetSupportedTargetsForTest(requested);
    }
    if (argc == 2 && std::string(argv[1]) == "verify") return verify();
    if (argc < 6) {
        std::fprintf(stderr, "usage: bench_bm_kernels verify | ssd BLOCK VARIANT COUNT ITERS [GRAY8] | dct BLOCK VARIANT COUNT ITERS [pair|forward|inverse] [GRAY8]\n");
        return 2;
    }
    const std::string mode = argv[1];
    const int block = std::atoi(argv[2]), variant = std::atoi(argv[3]), count = std::atoi(argv[4]);
    const int iterations = std::atoi(argv[5]);
    if (((mode == "ssd" && block != 12 && block != 16) || (mode == "dct" && block != 4 && block != 8 && block != 12 && block != 16)) || variant < 0 || variant > (mode == "dct" && block == 16 ? 4 : 2) || count < 1 || count > 129 || iterations < 1)
        return 2;
    double seconds = 0, checksum = 0;
    if (mode == "ssd") {
        auto kernel = ssd(block, variant);
        if (!kernel) return 77;
        const int stride = 1920, height = 1080;
        auto source = noise(stride * height);
        if (argc > 6) {
            std::ifstream input(argv[6], std::ios::binary);
            std::vector<unsigned char> raw(source.size());
            if (!input.read(reinterpret_cast<char*>(raw.data()), raw.size())) return 2;
            for (std::size_t i = 0; i < source.size(); ++i) source[i] = raw[i] / 255.f + source[i] * (3.f / 255.f);
        }
        // Cycle through a full-frame query corpus; do not benchmark one cached patch.
        struct Job { const float* ref; const float* candidates; };
        std::vector<Job> jobs;
        for (int y = 8; y < height - block - 8; y += 16)
            for (int x = 8; x < stride - block - count - 8; x += 16)
                jobs.push_back({source.data() + y * stride + x + count / 2,
                                source.data() + (y - 7) * stride + x});
        std::vector<float> out(count);
        for (const auto& job : jobs) kernel(job.ref, job.candidates, stride, count, out.data());
        auto start = Clock::now();
        std::size_t j = 0;
        for (int i = 0; i < iterations; ++i) {
            kernel(jobs[j].ref, jobs[j].candidates, stride, count, out.data());
            checksum += out[i % count];
            if (++j == jobs.size()) j = 0;
        }
        seconds = std::chrono::duration<double>(Clock::now() - start).count();
    } else if (mode == "dct") {
        auto kernel = nss::detail::dct_lab_kernel(block, variant);
        if (!kernel) return 77;
        auto patches = noise(static_cast<std::size_t>(count) * block * block);
        const std::string direction = argc > 6 ? argv[6] : "pair";
        if (direction != "pair" && direction != "forward" && direction != "inverse") return 2;
        if (argc > 7 || direction != "pair") {
            // Refresh a ring outside each timed interval; no cumulative transform
            // drift in one-direction measurements. Corpus construction is untimed.
            constexpr int ring = 64;
            auto input = noise(patches.size() * ring);
            if (argc > 7) {
                std::ifstream file(argv[7], std::ios::binary);
                std::vector<unsigned char> raw(1920 * 1080);
                if (!file.read(reinterpret_cast<char*>(raw.data()), raw.size())) return 2;
                for (std::size_t p = 0; p < input.size() / (block * block); ++p) {
                    const int x = int(p * 37) % (1920 - block + 1), y = int(p * 19) % (1080 - block + 1);
                    for (int r = 0; r < block; ++r)
                      for (int c = 0; c < block; ++c) {
                        const auto i = p * block * block + r * block + c;
                        input[i] = raw[(y + r) * 1920 + x + c] / 255.f + input[i] * (3.f / 255.f);
                      }
                }
            }
            std::vector<float> work(input.size());
            for (int done = 0; done < iterations; done += ring) {
                std::copy(input.begin(), input.end(), work.begin());
                const int n = std::min(ring, iterations - done);
                auto start = Clock::now();
                for (int j = 0; j < n; ++j) {
                    float* p = work.data() + j * patches.size();
                    if (direction != "inverse") kernel(p, count, false);
                    if (direction != "forward") kernel(p, count, true);
                }
                seconds += std::chrono::duration<double>(Clock::now() - start).count();
                checksum += work[(done / ring) % work.size()];
            }
        } else {
            kernel(patches.data(), count, false);
            kernel(patches.data(), count, true);
            auto start = Clock::now();
            for (int i = 0; i < iterations; ++i) {
                kernel(patches.data(), count, false);
                kernel(patches.data(), count, true);
            }
            seconds = std::chrono::duration<double>(Clock::now() - start).count();
        }
        for (float value : patches) checksum += value;
    } else return 2;
    const char* direction = mode == "dct" ? (argc > 6 ? argv[6] : "pair") : "ssd";
    const char* workload = mode == "ssd" ? "query_corpus" : (argc > 7 ? "patch_corpus" : (argc > 6 && std::string(argv[6]) != "pair" ? "fresh_ring" : "resident"));
    if (!std::isfinite(checksum)) return 1;
    std::printf("{\"mode\":\"%s\",\"direction\":\"%s\",\"workload\":\"%s\",\"block\":%d,\"variant\":%d,\"count\":%d,\"iterations\":%d,\"seconds\":%.9f,\"ns_per_item\":%.6f,\"checksum\":%.9g}\n",
                mode.c_str(), direction, workload, block, variant, count, iterations, seconds,
                seconds * 1e9 / (double(iterations) * count * (mode == "dct" && (argc <= 6 || std::string(argv[6]) == "pair") ? 2 : 1)), checksum);
}
