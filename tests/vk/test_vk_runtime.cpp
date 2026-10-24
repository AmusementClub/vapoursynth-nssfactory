// SPDX-License-Identifier: GPL-2.0-only
// V2 runtime test: budget charging, plane round trips through BAR and staging
// uploads, slots chained on timeline semaphores, frames abandoned by an
// exception, frames in flight under a memory limit, create/free without
// growth, the on-disk pipeline cache, timestamps and the process-lifetime device.
// Exits 77 when no device is supported; device creation errors fail.
//
// usage: test_vk_runtime [device_id]   (an explicit id may name an
// unsupported device, e.g. lavapipe, for a functional run)
#include "nss/resources.hpp"
#include "vulkan/runtime/context.hpp"
#include "vulkan/runtime/instance.hpp"
#include "vulkan/runtime/memory.hpp"
#include "vulkan/runtime/pipeline.hpp"
#include "vulkan/runtime/slot.hpp"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace nss_vk;

int failures = 0;

void expect(bool condition, const std::string& what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

// UTF-8 text of a path (std::filesystem::path::string() may throw on Windows).
std::string text(const std::filesystem::path& path) {
    const std::u8string utf8 = path.u8string();
    return std::string(utf8.begin(), utf8.end());
}

void set_env(const char* name, const std::filesystem::path& value) {
#ifdef _WIN32
    const std::wstring wide(name, name + std::strlen(name));
    _wputenv_s(wide.c_str(), value.wstring().c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

struct Push {
    VkDeviceAddress src;
    VkDeviceAddress dst;
    std::uint32_t width, height, src_pitch, dst_pitch;
    float a, b;
    std::uint32_t repeat;
};

constexpr int kWidth = 1923;  // odd sizes: partial workgroups, padded pitch
constexpr int kHeight = 1081;
constexpr std::ptrdiff_t kHostStride = 1931 * sizeof(float);
constexpr std::size_t kRowBytes = kWidth * sizeof(float);
constexpr std::size_t kPitch = (kRowBytes + 255) / 256 * 256;

// One execution slot of the test "filter": a slot and its planes.
struct Work {
    Slot slot;
    UploadPlane in;
    DownloadPlane out;
    Work(const std::shared_ptr<Device>& device, const std::shared_ptr<nss::ResourceBudget>& budget,
         UploadPlane::Mode mode, std::uint32_t timestamps = 0)
        : slot(device, timestamps), in(device, kPitch, kHeight, budget, mode), out(device, kPitch, kHeight, budget) {}
    void settle() noexcept { slot.settle(); }
};

std::vector<float> host_plane(float seed) {
    std::vector<float> plane(static_cast<std::size_t>(kHostStride / sizeof(float)) * kHeight, -1.0f);
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            plane[static_cast<std::size_t>(y) * (kHostStride / sizeof(float)) + x] =
                std::sin(0.001f * static_cast<float>(x * 7 + y * 13)) * 100.0f + seed;
        }
    }
    return plane;
}

// Uploads `input`, runs out = 2 * in + b on the slot, downloads into `output`.
void run_frame(Work& w, const Pipeline& pipeline, const std::vector<float>& input, float b, std::vector<float>& output,
               bool label = false) {
    w.in.write(input.data(), kHostStride, kRowBytes);
    if (w.in.needs_copy()) w.in.record_upload(w.slot.commands(Slot::Upload));
    if (label) w.slot.begin_label(Slot::Compute, "plane_scale");
    w.slot.mark(Slot::Compute, 0);
    const Push push{w.in.address(), w.out.address(), kWidth, kHeight,
                    static_cast<std::uint32_t>(kPitch / sizeof(float)), static_cast<std::uint32_t>(kPitch / sizeof(float)),
                    2.0f, b, 1};
    pipeline.dispatch(w.slot.commands(Slot::Compute), &push, (kWidth + 15) / 16, (kHeight + 15) / 16);
    w.slot.mark(Slot::Compute, 1);
    if (label) w.slot.end_label(Slot::Compute);
    w.out.record_download(w.slot.commands(Slot::Download));
    w.slot.submit();
    w.slot.wait();
    output.assign(input.size(), 0.0f);
    w.out.read(output.data(), kHostStride, kRowBytes);
}

// 2 * x + b is exact in the product, so fused and unfused drivers agree.
bool check_frame(const std::vector<float>& input, float b, const std::vector<float>& output) {
    const std::size_t stride = kHostStride / sizeof(float);
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            const std::size_t i = static_cast<std::size_t>(y) * stride + x;
            if (output[i] != 2.0f * input[i] + b) {
                std::fprintf(stderr, "mismatch at (%d, %d): %.9g != %.9g\n", x, y, output[i], 2.0f * input[i] + b);
                return false;
            }
        }
    }
    return true;
}

struct Live {
    std::uint32_t allocations;
    VkDeviceSize block_bytes;
};
Live live(const Device& device) {
    VmaTotalStatistics stats{};
    vmaCalculateStatistics(device.allocator(), &stats);
    return {stats.total.statistics.allocationCount, stats.total.statistics.blockBytes};
}

}  // namespace

int main(int argc, char** argv) {
    std::error_code ignored;
    const std::filesystem::path cache_dir =
        std::filesystem::temp_directory_path(ignored) / ("nss_vk_cache_test_" + std::to_string(std::random_device{}()));
    set_env("NSS_VK_PIPELINE_CACHE", cache_dir);

    const Instance& vk = instance();
    if (vk.devices.empty()) {
        std::printf("vk runtime SKIP: no Vulkan device (%s)\n", vk.error.c_str());
        return 77;
    }
    // Skip only when no device is supported; a supported device that fails
    // to open is a failure.
    int index = -1;
    if (argc > 1) {
        index = std::atoi(argv[1]);
    } else {
        for (int i = 0; i < static_cast<int>(vk.devices.size()) && index < 0; ++i) {
            if (classify(query_caps(vk.devices[static_cast<std::size_t>(i)])).support != Support::unsupported) index = i;
        }
    }
    if (index < 0) {
        std::printf("vk runtime SKIP: no supported Vulkan device\n");
        return 77;
    }
    std::shared_ptr<Device> device;
    try {
        device = Device::open(index, argc > 1);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: opening device %d: %s\n", index, error.what());
        return EXIT_FAILURE;
    }
    std::printf("device %d: %s, support %s, rebar %d, queues compute %u/%u upload %u/%u download %u/%u\n",
                device->index(), device->label().c_str(), support_name(device->support().support), device->rebar(),
                device->compute().family, device->compute().index, device->upload().family, device->upload().index,
                device->download().family, device->download().index);
    const Live baseline = live(*device);

    // Budget: charges, overrun error, release.
    {
        auto budget = std::make_shared<nss::ResourceBudget>(64u << 20);
        Buffer a(device, 48u << 20, Memory::Device, budget);
        bool overrun = false;
        try {
            Buffer b(device, 32u << 20, Memory::Staging, budget);
        } catch (const std::runtime_error& error) {
            overrun = std::strstr(error.what(), "memory_limit_mb exceeded") != nullptr;
        }
        expect(overrun, "budget overrun is reported as memory_limit_mb exceeded");
        expect(budget->snapshot().owned == (48u << 20), "failed allocation leaves no charge");
        a.reset();
        expect(budget->snapshot().owned == 0, "reset releases the charge");
        Buffer c(device, 64u << 20, Memory::Readback, budget);
        expect(budget->snapshot().bytes[static_cast<std::size_t>(nss::ResourceKind::Pinned)] == (64u << 20),
               "host memory is charged as Pinned");
        expect(c.mapped() != nullptr && c.address() != 0, "readback buffer is mapped and addressable");
    }

    Pipeline pipeline(device, {"runtime_plane_scale", sizeof(Push), 0, {}});
    device->save_pipeline_cache();
    expect(std::filesystem::exists(device->pipeline_cache_file(), ignored) &&
               std::filesystem::file_size(device->pipeline_cache_file(), ignored) > 0,
           "pipeline cache written to " + text(device->pipeline_cache_file()));

    // Round trips: BAR (when the device has it) and staging.
    for (UploadPlane::Mode mode : {UploadPlane::Mode::automatic, UploadPlane::Mode::staging}) {
        Work w(device, nullptr, mode, 2);
        const std::vector<float> input = host_plane(3.0f);
        std::vector<float> output;
        run_frame(w, pipeline, input, 1.0f, output, true);
        const char* path = w.in.needs_copy() ? "staging" : "BAR";
        expect(check_frame(input, 1.0f, output), std::string("round trip through ") + path);
        // The host-side padding of the output rows is never written.
        expect(output[kRowBytes / sizeof(float)] == 0.0f, "row padding untouched");
        const std::vector<double> times = w.slot.timestamps_ns();
        if (device->caps().host_query_reset && device->compute().timestamp_bits) {
            expect(times.size() == 2 && times[1] > 0.0, "kernel timestamps");
            std::printf("plane_scale %s: %.1f us\n", path, times[1] / 1000.0);
        }
        // More frames on the same slot (pools reset, semaphore continues,
        // queries reset by every wait).
        for (int f = 0; f < 3; ++f) {
            run_frame(w, pipeline, input, 5.0f + f, output);
            expect(check_frame(input, 5.0f + f, output), std::string("next frame through ") + path);
        }
    }

    // A frame abandoned while recording, and one submitted but never waited
    // for: returning the lease settles the slot and the next frame is clean.
    {
        std::vector<std::unique_ptr<Work>> one;
        one.push_back(std::make_unique<Work>(device, nullptr, UploadPlane::Mode::automatic));
        SlotPool<Work> pool(std::move(one));
        const std::vector<float> input = host_plane(2.0f);
        std::vector<float> output;
        try {
            auto lease = pool.acquire();
            const Push push{lease->in.address(), lease->out.address(), kWidth, kHeight, 0, 0, 7.0f, 7.0f, 1};
            pipeline.dispatch(lease->slot.commands(Slot::Compute), &push, 1, 1);  // pitch 0: garbage on purpose
            throw std::runtime_error("abandoned");
        } catch (const std::runtime_error&) {
        }
        {
            auto lease = pool.acquire();
            lease->in.write(input.data(), kHostStride, kRowBytes);
            if (lease->in.needs_copy()) lease->in.record_upload(lease->slot.commands(Slot::Upload));
            lease->slot.submit();  // pending when the lease ends
        }
        auto lease = pool.acquire();
        run_frame(*lease, pipeline, input, 4.0f, output);
        expect(check_frame(input, 4.0f, output), "frame after abandoned frames");
    }

    // Frames in flight: 3 slots, 8 threads, under a limit that fits exactly 3.
    {
        const std::size_t plane = kPitch * kHeight;
        const std::size_t per_slot = device->rebar() ? 3 * plane : 4 * plane;
        auto budget = std::make_shared<nss::ResourceBudget>(3 * per_slot);
        std::vector<std::unique_ptr<Work>> slots;
        try {
            for (int i = 0; i < 3; ++i) {
                slots.push_back(std::make_unique<Work>(device, budget, UploadPlane::Mode::automatic));
            }
        } catch (const std::exception& error) {
            // E.g. BAR exhausted: the staging fallback needs a fourth plane per slot.
            std::fprintf(stderr, "FAIL: three slots within their limit: %s\n", error.what());
            return EXIT_FAILURE;
        }
        bool fourth = false;
        try {
            Work extra(device, budget, UploadPlane::Mode::automatic);
        } catch (const std::runtime_error& error) {
            fourth = std::strstr(error.what(), "memory_limit_mb exceeded") != nullptr;
        }
        expect(fourth, "a fourth slot exceeds the limit sized for three");
        SlotPool<Work> pool(std::move(slots));
        std::atomic<int> bad{0};
        std::vector<std::thread> threads;
        for (int t = 0; t < 8; ++t) {
            threads.emplace_back([&, t] {
                std::vector<float> output;
                for (int f = 0; f < 12; ++f) {
                    const float b = static_cast<float>(t * 100 + f);
                    const std::vector<float> input = host_plane(static_cast<float>(f));
                    try {
                        auto lease = pool.acquire();
                        run_frame(*lease, pipeline, input, b, output);
                    } catch (const std::exception& error) {
                        std::fprintf(stderr, "thread %d frame %d: %s\n", t, f, error.what());
                        ++bad;
                        continue;
                    }
                    if (!check_frame(input, b, output)) ++bad;
                }
            });
        }
        for (std::thread& thread : threads) thread.join();
        expect(bad == 0, "96 frames on 3 slots from 8 threads");
        expect(budget->snapshot().peak <= 3 * per_slot, "peak within the limit");
    }

    // Create/free: no growth in live allocations.
    for (int i = 0; i < 100; ++i) {
        auto budget = std::make_shared<nss::ResourceBudget>(SIZE_MAX);
        Work w(device, budget, i % 2 ? UploadPlane::Mode::staging : UploadPlane::Mode::automatic);
        Pipeline p(device, {"runtime_plane_scale", sizeof(Push), 0, {}});
        if (i % 25 == 0) {
            const std::vector<float> input = host_plane(1.0f);
            std::vector<float> output;
            run_frame(w, p, input, 0.0f, output);
            expect(check_frame(input, 0.0f, output), "create/free round trip");
        }
    }
    pipeline.reset();
    const Live after = live(*device);
    expect(after.allocations == baseline.allocations && after.block_bytes == baseline.block_bytes,
           "no allocations or blocks left after create/free (" + std::to_string(after.allocations) + " / " +
               std::to_string(after.block_bytes) + " vs " + std::to_string(baseline.allocations) + " / " +
               std::to_string(baseline.block_bytes) + ")");

    // The device lives for the process: released by every filter, opening it
    // again returns the same one, still usable.
    const Device* first = device.get();
    device.reset();
    try {
        device = Device::open(index, argc > 1);
        expect(device.get() == first, "the device is kept for the process");
        Work w(device, nullptr, UploadPlane::Mode::automatic);
        Pipeline p(device, {"runtime_plane_scale", sizeof(Push), 0, {}});
        const std::vector<float> input = host_plane(6.0f);
        std::vector<float> output;
        run_frame(w, p, input, 2.0f, output);
        expect(check_frame(input, 2.0f, output), "frame on a reopened device");
    } catch (const std::exception& error) {
        expect(false, std::string("reopening the device: ") + error.what());
    }

    std::filesystem::remove_all(cache_dir, ignored);
    if (failures) return EXIT_FAILURE;
    std::printf("vk runtime OK\n");
    return EXIT_SUCCESS;
}
