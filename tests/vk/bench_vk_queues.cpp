// SPDX-License-Identifier: GPL-2.0-only
// Frame throughput of the D28 queue layouts (not a ctest): staged or BAR
// uploads of a 1080p plane, a kernel of a chosen length, a download, on
// several slots from several threads. Compare layouts with NSS_VK_COPY_QUEUES
// (0, 1, 2), NSS_VK_ONE_COPY_QUEUE=shared and NSS_VK_REBAR=0.
//
// usage: bench_vk_queues [device_id] [repeat] [frames] [slots] [threads]
#include "vulkan/runtime/context.hpp"
#include "vulkan/runtime/memory.hpp"
#include "vulkan/runtime/pipeline.hpp"
#include "vulkan/runtime/slot.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

namespace {

using namespace nss_vk;

struct Push {
    VkDeviceAddress src;
    VkDeviceAddress dst;
    std::uint32_t width, height, src_pitch, dst_pitch;
    float a, b;
    std::uint32_t repeat;
};

constexpr int kWidth = 1920, kHeight = 1080;
constexpr std::size_t kRowBytes = kWidth * sizeof(float);

struct Work {
    Slot slot;
    UploadPlane in;
    DownloadPlane out;
    explicit Work(const std::shared_ptr<Device>& device)
        : slot(device, 2), in(device, kRowBytes, kHeight), out(device, kRowBytes, kHeight) {}
    void settle() noexcept { slot.settle(); }
};

void frame(Work& w, const Pipeline& pipeline, const std::vector<float>& input, std::vector<float>& output,
           std::uint32_t repeat) {
    w.in.write(input.data(), kRowBytes, kRowBytes);
    if (w.in.needs_copy()) w.in.record_upload(w.slot.commands(Slot::Upload));
    w.slot.mark(Slot::Compute, 0);
    const Push push{w.in.address(), w.out.address(), kWidth, kHeight, kWidth, kWidth, 2.0f, 1.0f, repeat};
    pipeline.dispatch(w.slot.commands(Slot::Compute), &push, kWidth / 16, (kHeight + 15) / 16);
    w.slot.mark(Slot::Compute, 1);
    w.out.record_download(w.slot.commands(Slot::Download));
    w.slot.submit();
    w.slot.wait();
    w.out.read(output.data(), kRowBytes, kRowBytes);
}

}  // namespace

int main(int argc, char** argv) {
    const int index = argc > 1 ? std::atoi(argv[1]) : 0;
    const std::uint32_t repeat = argc > 2 ? static_cast<std::uint32_t>(std::atoi(argv[2])) : 1;
    const int frames = argc > 3 ? std::atoi(argv[3]) : 600;
    const int slots = argc > 4 ? std::atoi(argv[4]) : 3;
    const int threads = argc > 5 ? std::atoi(argv[5]) : 6;
    std::shared_ptr<Device> device = Device::open(index, true);
    const Pipeline pipeline(device, {"runtime_plane_scale", sizeof(Push), 0, {}});
    std::vector<std::unique_ptr<Work>> works;
    for (int i = 0; i < slots; ++i) works.push_back(std::make_unique<Work>(device));
    const bool staged = works[0]->in.needs_copy();
    SlotPool<Work> pool(std::move(works));

    std::vector<float> input(static_cast<std::size_t>(kWidth) * kHeight, 0.5f), output(input.size());
    double kernel_ms = 0;
    for (int i = 0; i < 20; ++i) {  // warm up; kernel time from the last
        auto lease = pool.acquire();
        frame(*lease, pipeline, input, output, repeat);
        kernel_ms = lease->slot.timestamps_ns()[1] / 1e6;
    }
    std::atomic<int> next{0};
    const auto start = std::chrono::steady_clock::now();
    std::vector<std::thread> pool_threads;
    for (int t = 0; t < threads; ++t) {
        pool_threads.emplace_back([&] {
            std::vector<float> in(input), out(input.size());
            while (next.fetch_add(1) < frames) {
                auto lease = pool.acquire();
                frame(*lease, pipeline, in, out, repeat);
            }
        });
    }
    for (std::thread& t : pool_threads) t.join();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("queues compute %u/%u upload %u/%u download %u/%u, %s uploads, kernel %.2f ms: %.1f fps\n",
                device->compute().family, device->compute().index, device->upload().family, device->upload().index,
                device->download().family, device->download().index, staged ? "staged" : "BAR", kernel_ms,
                frames / seconds);
    return 0;
}
