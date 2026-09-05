#include "cpu/bm/kernel_lab.hpp"

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

static void reference_dct2(std::vector<float>& data, bool inverse) {
    constexpr double pi = 3.14159265358979323846;
    double table[16][16];
    for (int k = 0; k < 16; ++k)
        for (int i = 0; i < 16; ++i)
            table[k][i] = std::sqrt((k == 0 ? 1.0 : 2.0) / 16.0) * std::cos(pi * (i + .5) * k / 16.0);
    for (std::size_t p = 0; p < data.size(); p += 256) {
        double tmp[256];
        for (int r = 0; r < 16; ++r)
            for (int c = 0; c < 16; ++c) {
                double sum = 0;
                for (int i = 0; i < 16; ++i)
                    sum += (inverse ? table[i][r] : table[r][i]) * data[p + i * 16 + c];
                tmp[r * 16 + c] = sum;
            }
        for (int r = 0; r < 16; ++r)
            for (int c = 0; c < 16; ++c) {
                double sum = 0;
                for (int i = 0; i < 16; ++i)
                    sum += (inverse ? table[i][c] : table[c][i]) * tmp[r * 16 + i];
                data[p + r * 16 + c] = static_cast<float>(sum);
            }
    }
}

static int verify() {
    if (!ssd(12, 0) || !nss::detail::dct16_lab_kernel(0)) return 77;
    int failed = 0;
#if defined(__unix__)
    // End the final candidate patch at a PROT_NONE page. Masked/tail loads
    // must not read beyond the row even when no padding is available.
    const std::size_t page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    for (int b : {12, 16}) {
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
                ssd(b, v)(source, source, stride, count, output.data());
                if (std::memcmp(reference.data(), output.data(), count * sizeof(float))) ++failed;
            }
            munmap(mapping, (pages + 1) * page);
        }
    }
    std::printf("{\"verify\":\"ssd_guard_pages\",\"failed\":%d}\n", failed);
#endif
    for (int b : {12, 16}) {
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
    for (int v = 0; v < 5; ++v) {
        auto kernel = nss::detail::dct16_lab_kernel(v);
        std::size_t different = 0;
        double max_error = 0, roundtrip = 0;
        for (int count : {1, 8, 32}) {
            for (int pattern = 0; pattern < 4; ++pattern) {
                auto input = noise(static_cast<std::size_t>(count) * 256);
                if (pattern == 1) std::fill(input.begin(), input.end(), 1.f);
                if (pattern == 2) { std::fill(input.begin(), input.end(), 0.f); input[17] = 1.f; }
                if (pattern == 3)
                    for (std::size_t i = 0; i < input.size(); ++i) input[i] = float((i / 16) % 16) / 16.f;
                for (bool inverse : {false, true}) {
                    auto baseline = input, got = input, reference = input;
                    nss::detail::dct16_lab_kernel(0)(baseline.data(), count, inverse);
                    if (!kernel(got.data(), count, inverse)) ++failed;
                    reference_dct2(reference, inverse);
                    for (std::size_t i = 0; i < got.size(); ++i) {
                        different += std::memcmp(&baseline[i], &got[i], sizeof(float)) != 0;
                        max_error = std::max(max_error, std::abs(double(reference[i]) - got[i]));
                    }
                }
                auto got = input;
                kernel(got.data(), count, false);
                kernel(got.data(), count, true);
                for (std::size_t i = 0; i < got.size(); ++i)
                    roundtrip = std::max(roundtrip, std::abs(double(input[i]) - got[i]));
            }
        }
        std::printf("{\"verify\":\"dct\",\"variant\":%d,\"different\":%zu,\"reference_max\":%.9g,\"roundtrip_max\":%.9g}\n",
                    v, different, max_error, roundtrip);
        if (different || max_error > 2e-5 || roundtrip > 2e-5) ++failed;
    }
    return failed ? 1 : 0;
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "verify") return verify();
    if (argc < 6) {
        std::fprintf(stderr, "usage: bench_bm_kernels verify | ssd BLOCK VARIANT COUNT ITERS [GRAY8] | dct 16 VARIANT COUNT ITERS\n");
        return 2;
    }
    const std::string mode = argv[1];
    const int block = std::atoi(argv[2]), variant = std::atoi(argv[3]), count = std::atoi(argv[4]);
    const int iterations = std::atoi(argv[5]);
    if ((block != 12 && block != 16) || variant < 0 || variant > (mode == "dct" ? 4 : 2) || count < 1 || count > 129 || iterations < 1)
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
    } else if (mode == "dct" && block == 16) {
        auto kernel = nss::detail::dct16_lab_kernel(variant);
        if (!kernel) return 77;
        auto patches = noise(static_cast<std::size_t>(count) * 256);
        auto start = Clock::now();
        for (int i = 0; i < iterations; ++i) {
            kernel(patches.data(), count, false);
            kernel(patches.data(), count, true);
        }
        seconds = std::chrono::duration<double>(Clock::now() - start).count();
        for (float value : patches) checksum += value;
    } else return 2;
    std::printf("{\"mode\":\"%s\",\"block\":%d,\"variant\":%d,\"count\":%d,\"iterations\":%d,\"seconds\":%.9f,\"ns_per_item\":%.6f,\"checksum\":%.9g}\n",
                mode.c_str(), block, variant, count, iterations, seconds,
                seconds * 1e9 / (double(iterations) * count * (mode == "dct" ? 2 : 1)), checksum);
}
